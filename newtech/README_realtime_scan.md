# realtime_scan_minecraft

MediaPipeで手の動きを取り、輪郭をMinecraftにブロック再現するサンプルです。

## できること
- 親指 + 人差し指のピンチで輪郭トレース
- サムズアップ中に手を上下して拡大縮小
- 手を開くと現在の輪郭をMinecraftへ再構築
- 握りこぶしで輪郭と配置済みブロックをクリア
- `B` キーで強制再構築（ジェスチャー不要）
- `C` キーで配置済みブロックだけクリア

## 事前準備
```powershell
python -m pip install mediapipe opencv-python numpy minecraft-remote-api
```

## 実行
```powershell
$env:MC_REMOTE_HOST='mc-remote.xgames.jp'
$env:MC_REMOTE_PORT='25575'
$env:MC_REMOTE_PLAYER_NAME='YOUR_PLAYER_NAME'
$env:MC_REMOTE_ORIGIN_X='1000'
$env:MC_REMOTE_ORIGIN_Y='100'
$env:MC_REMOTE_ORIGIN_Z='1000'
$env:MC_REMOTE_BLOCK='lime_concrete'
python newtech\realtime_scan_minecraft.py
```

## 補足
- 輪郭は `MC_REMOTE_ORIGIN_X/Y/Z` を基準に X-Y 平面へ生成されます。
- スキャン精度は `PIXELS_PER_BLOCK` と `MIN_TRACE_DISTANCE_PX` で調整できます。
- 画面左上 `MC Remote: Connected` になっていない場合はMinecraft側未接続です。
