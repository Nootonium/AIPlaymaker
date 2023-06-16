# config.py
import os


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")

    # ...


class Development(Config):
    DEBUG = True
    TESTING = True


class Staging(Config):
    DEBUG = False
    TESTING = True
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = False


class Production(Config):
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
