# 🌐 NeuroServe API Guide

This file explains how to interact with **NeuroServe** APIs (core services + plugins).

---

## 🔑 Authentication
- Currently, **no authentication system** is required (open access).
- In the future, **API Key** or **JWT** support may be added.

---

## 🚀 Base URLs
- Local: `http://127.0.0.1:8000`
- Deployment: `https://your-domain.com`

---

## 🧩 Core Services
### ✅ Health Check
- **GET** `/health`
- **Description:** Verify that the server is running.
- **Response:**
```json
{"status": "ok"}
```
