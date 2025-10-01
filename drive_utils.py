# -*- coding: utf-8 -*-
"""Google Drive helpers for Streamlit (Service Account).
Includes upload/download for RAG cache files.
ASCII-safe, Python 3.8+ compatible.
"""
import io
import os
import time

import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

SCOPES = [
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource(show_spinner=False)
def authenticate_drive():
    """Authenticate using a service account JSON from Streamlit secrets.
    Expect secrets.GOOGLE_SERVICE_ACCOUNT_JSON (string or dict-like).
    """
    sa_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", None)
    if not sa_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is missing in secrets.")

    if isinstance(sa_json, str):
        import json
        try:
            sa_info = json.loads(sa_json)
        except Exception as e:
            raise RuntimeError("Invalid GOOGLE_SERVICE_ACCOUNT_JSON: %s" % e)
    elif isinstance(sa_json, dict):
        sa_info = sa_json
    else:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON must be a JSON string or dict.")

    # Normalize private_key newlines
    if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
        pk = sa_info["private_key"]
        # keep as raw with literal \n then decode escapes to real newlines
        pk = pk.replace("\\n", "\n")
        pk = pk.encode("utf-8").decode("unicode_escape")
        sa_info["private_key"] = pk

    creds = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return service


def _retry(callable_fn, max_tries=3, base_delay=0.8):
    last_err = None
    for i in range(max_tries):
        try:
            return callable_fn()
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** i))
    if last_err:
        raise last_err


def format_file_size(size_str):
    if not size_str:
        return "-"
    try:
        size = int(size_str)
    except Exception:
        return str(size_str)
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    val = float(size)
    while val >= 1024.0 and i < len(units) - 1:
        val = val / 1024.0
        i += 1
    if i == 0:
        return "%d %s" % (val, units[i])
    return "%.1f %s" % (val, units[i])


def list_files_in_folder(service, folder_id):
    files = []
    page_token = None
    q = "'%s' in parents and trashed = false" % folder_id
    fields = "nextPageToken, files(id,name,size,mimeType,modifiedTime)"
    while True:
        def _call():
            return service.files().list(q=q, pageToken=page_token, fields=fields, orderBy="name").execute()
        resp = _retry(_call)
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _find_file_by_name(service, folder_id, filename):
    safe_name = filename.replace("'", "\'")
    q = "'%s' in parents and name = '%s' and trashed = false" % (folder_id, safe_name)
    fields = "files(id,name)"
    def _call():
        return service.files().list(q=q, fields=fields, pageSize=1).execute()
    resp = _retry(_call)
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    return None


def download_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        def _step():
            return downloader.next_chunk()
        status, done = _retry(_step)
    fh.seek(0)
    return fh


def upload_file(service, folder_id, local_path, mime_type="application/octet-stream"):
    filename = os.path.basename(local_path)
    file_id = _find_file_by_name(service, folder_id, filename)
    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)

    if file_id:
        try:
            def _upd():
                return service.files().update(fileId=file_id, media_body=media, fields="id").execute()
            _retry(_upd)
            return file_id
        except HttpError:
            pass

    file_metadata = {"name": filename, "parents": [folder_id]}
    def _create():
        return service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    created = _retry(_create)
    return created["id"]


# ====== Convenience helpers for RAG caches ======

def download_embeddings_from_drive(service, folder_id, embeddings_name="embeddings_meta.pkl", faiss_name="faiss_index.bin"):
    """Try to download cache files from Drive into current working dir.
    Returns a dict with local paths if found.
    """
    out = {"embeddings_path": None, "faiss_path": None}

    emb_id = _find_file_by_name(service, folder_id, embeddings_name)
    if emb_id:
        buf = download_file(service, emb_id)
        with open(embeddings_name, "wb") as f:
            f.write(buf.getvalue())
        out["embeddings_path"] = os.path.abspath(embeddings_name)

    faiss_id = _find_file_by_name(service, folder_id, faiss_name)
    if faiss_id:
        buf = download_file(service, faiss_id)
        with open(faiss_name, "wb") as f:
            f.write(buf.getvalue())
        out["faiss_path"] = os.path.abspath(faiss_name)

    return out


def upload_embeddings_to_drive(service, folder_id, embeddings_path="embeddings_meta.pkl", faiss_path="faiss_index.bin"):
    """Upload (or update) cache files to Drive folder. Returns dict of file ids."
    """
    out = {"embeddings_id": None, "faiss_id": None}
    if os.path.exists(embeddings_path):
        out["embeddings_id"] = upload_file(service, folder_id, embeddings_path, mime_type="application/octet-stream")
    if os.path.exists(faiss_path):
        out["faiss_id"] = upload_file(service, folder_id, faiss_path, mime_type="application/octet-stream")
    return out
