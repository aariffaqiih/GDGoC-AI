#!/usr/bin/env python3
"""
Inisialisasi database secara manual.
Jalankan dengan `python init_db_cli.py` sebelum menjalankan aplikasi jika diperlukan.
"""

import sys
import logging
from database import init_db
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        init_db(Config.DB_PATH)
        logger.info("Database berhasil diinisialisasi di %s", Config.DB_PATH)
    except Exception as e:
        logger.critical("Gagal inisialisasi database: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()