# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['..\\run.py'],
    pathex=[],
    binaries=[],
    datas=[('c:\\Users\\bry15\\jupyter Projects\\视频分析技术\\flood-watch\\frontend', 'frontend')],
    hiddenimports=['uvicorn.logging', 'uvicorn.loops.auto', 'uvicorn.loops.asyncio', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.http.h11_impl', 'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on', 'uvicorn.lifespan.off'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'ultralytics', 'PIL', 'aiohttp', 'requests', 'bs4', 'soupsieve', 'cryptography', 'matplotlib', 'scipy', 'pandas', 'lxml', 'IPython', 'tkinter', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FloodWatch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FloodWatch',
)
