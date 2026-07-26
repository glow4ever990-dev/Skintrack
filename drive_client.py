# -*- coding: utf-8 -*-
"""
Google Drive 访问层（服务账号方式）
只读取照片、只更新一个结果文件，不改动你的其他文件。
"""

import io
import re

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff")


class DriveClient:
    def __init__(self, sa_info: dict):
        self.creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=SCOPES
        )
        self.service = build("drive", "v3", credentials=self.creds,
                             cache_discovery=False)

    # ---------- 内部 ----------
    def _token(self):
        if not self.creds.valid:
            self.creds.refresh(Request())
        return self.creds.token

    # ---------- 列出照片 ----------
    def list_images(self, folder_id):
        """列出文件夹里所有图片，按创建时间排序"""
        files, token = [], None
        q = (f"'{folder_id}' in parents and mimeType contains 'image/' "
             f"and trashed = false")
        while True:
            resp = self.service.files().list(
                q=q,
                pageSize=200,
                pageToken=token,
                orderBy="createdTime",
                fields=("nextPageToken, files(id,name,createdTime,modifiedTime,"
                        "size,thumbnailLink,imageMediaMetadata(time,width,height))"),
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            files.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return files

    # ---------- 列出子文件夹（每人一个） ----------
    def list_subfolders(self, folder_id):
        """列出直接子文件夹，按名字排序。每个子文件夹当作一个人。"""
        folders, token = [], None
        q = (f"'{folder_id}' in parents "
             f"and mimeType = 'application/vnd.google-apps.folder' "
             f"and trashed = false")
        while True:
            resp = self.service.files().list(
                q=q, pageSize=200, pageToken=token, orderBy="name",
                fields="nextPageToken, files(id,name)",
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute()
            folders.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return folders

    def list_images_by_person(self, root_id):
        """
        扫描根文件夹下的每个子文件夹，返回 [(人名, 图片列表), ...]。
        直接丢在根目录里的图片归到"未分组"，兼容你之前的用法。
        """
        out = []
        loose = self.list_images(root_id)
        if loose:
            out.append(("未分组", loose))
        for f in self.list_subfolders(root_id):
            imgs = self.list_images(f["id"])
            if imgs:
                out.append((f["name"], imgs))
        return out

    # ---------- 下载图片字节 ----------
    def fetch_image_bytes(self, f, max_px=1400):
        """
        优先用 Drive 的缩略图接口取一张 max_px 的小图。
        这样 10MB 的原图不会整个进内存，速度也快很多。
        取不到缩略图才回退下载原图。
        """
        thumb = f.get("thumbnailLink")
        if thumb:
            url = re.sub(r"=s\d+(-c)?$", f"=s{max_px}", thumb)
            try:
                r = requests.get(url,
                                 headers={"Authorization": f"Bearer {self._token()}"},
                                 timeout=30)
                if r.ok and len(r.content) > 1000:
                    return r.content
            except Exception:
                pass

        # 回退：下载原图
        buf = io.BytesIO()
        req = self.service.files().get_media(fileId=f["id"], supportsAllDrives=True)
        from googleapiclient.http import MediaIoBaseDownload
        dl = MediaIoBaseDownload(buf, req, chunksize=1024 * 1024)
        done = False
        while not done:
            _, done = dl.next_chunk()
        return buf.getvalue()

    # ---------- 结果文件读写 ----------
    def find_file(self, folder_id, name):
        resp = self.service.files().list(
            q=(f"'{folder_id}' in parents and name = '{name}' and trashed = false"),
            fields="files(id,name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = resp.get("files", [])
        return files[0]["id"] if files else None

    def read_text_file(self, file_id):
        req = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buf = io.BytesIO()
        from googleapiclient.http import MediaIoBaseDownload
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        return buf.getvalue().decode("utf-8", errors="replace")

    def update_text_file(self, file_id, text):
        """
        更新一个已存在的文件（重要：服务账号自己没有存储配额，
        只能更新你手动建好的文件，不能新建。所以 README 里要求你
        先在文件夹里手动建一个空的 skin_metrics.csv）
        """
        media = MediaIoBaseUpload(
            io.BytesIO(text.encode("utf-8")), mimetype="text/csv", resumable=False
        )
        self.service.files().update(
            fileId=file_id, media_body=media, supportsAllDrives=True
        ).execute()
