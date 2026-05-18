#!/usr/bin/env python3
"""
Script untuk membuat user konselor
Jalankan: python create_user.py
"""

import sys
import os
import re
from getpass import getpass
from config import Config
from database import init_db, create_user, verify_connection

def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password minimal 8 karakter"
    if not re.search(r"[A-Z]", password):
        return False, "Password harus mengandung huruf kapital"
    if not re.search(r"[a-z]", password):
        return False, "Password harus mengandung huruf kecil"
    if not re.search(r"\d", password):
        return False, "Password harus mengandung angka"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password harus mengandung simbol (!@#$%^&* etc)"
    return True, ""

def main():
    # Inisialisasi database
    try:
        init_db(Config.DB_PATH)
        if not verify_connection():
            print("❌ Gagal terhubung ke database")
            sys.exit(1)
        print("✅ Database terhubung")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("\n=== Buat Akun Konselor StayLearn ===\n")
    
    username = input("Username: ").strip()
    if not username:
        print("❌ Username tidak boleh kosong")
        sys.exit(1)
    
    while True:
        password = getpass("Password: ").strip()
        if not password:
            print("❌ Password tidak boleh kosong")
            continue
        valid, msg = validate_password(password)
        if not valid:
            print(f"❌ {msg}")
            continue
        confirm = getpass("Konfirmasi Password: ").strip()
        if password != confirm:
            print("❌ Password tidak cocok")
            continue
        break
    
    success = create_user(username, password)
    if success:
        print(f"\n✅ Akun '{username}' berhasil dibuat!")
        print("   Silakan login di /login")
    else:
        print(f"\n❌ Username '{username}' sudah ada. Pilih username lain.")

if __name__ == "__main__":
    main()