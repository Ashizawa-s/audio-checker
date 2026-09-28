import os
import time
import uuid
import secrets
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from google import genai
import uvicorn

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("[WARN] 環境変数 GEMINI_API_KEY が設定されていません")
client = genai.Client(api_key=API_KEY)

app = FastAPI()
security = HTTPBasic()

USERNAME = os.environ.get("AUTH_USER", "admin")
PASSWORD = os.environ.get("AUTH_PASS", "password123")

# 優先して使うモデル（上から順に試す）。環境変数 GEMINI_MODELS でカンマ区切り上書き可
DEFAULT_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

# 音声監査に使えない特殊用途モデルを除外するためのキーワード
EXCLUDE_KEYWORDS = ("tts", "image", "live", "native-audio", "embedding", "omni", "computer-use")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="認証に失敗しました",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def get_target_models():
    """使うモデルの候補リストを作る。
    指定リスト（または環境変数）を優先し、APIから取得できた汎用flashモデルを後ろに補う。"""
    env_models = os.environ.get("GEMINI_MODELS")
    preferred = [m.strip() for m in env_models.split(",")] if env_models else list(DEFAULT_MODELS)

    available = []
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            # 新SDK(google-genai)では supported_actions。旧SDKの属性名にも一応対応
            actions = (getattr(m, "supported_actions", None)
                       or getattr(m, "supported_generation_methods", None) or [])
            if ("flash" in name.lower()
                    and "generateContent" in actions
                    and not any(k in name.lower() for k in EXCLUDE_KEYWORDS)):
                available.append(name)
    except Exception as e:
        # キーが無効な場合はここで分かるので、ログに原因を出しておく
        print(f"[WARN] モデル一覧の取得に失敗: {e}")

    if available:
        # 指定モデルのうち実在するものを先頭に、残りの取得モデルを後ろに
        models = [m for m in preferred if m in available]
        models += sorted([m for m in available if m not in models], reverse=True)
        return models
    return preferred


@app.get("/", response_class=HTMLResponse)
async def read_index(username: str = Depends(verify_credentials)):
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html が見つかりません</h1>"


@app.post("/analyze")
def analyze_audio(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    username: str = Depends(verify_credentials)
):
    # def（非async）にしたので、待機中に他のリクエストを止めない
    ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    temp_path = f"temp_{uuid.uuid4().hex}{ext}"
    audio_file = None

    try:
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        audio_file = client.files.upload(file=temp_path)

        # 長尺音声は処理に時間がかかるので待機時間を長めに
        max_wait = 180
        waited = 0
        while audio_file.state.name == "PROCESSING":
            if waited > max_wait:
                raise HTTPException(status_code=500, detail="音声ファイルの処理がタイムアウトしました。")
            time.sleep(3)
            waited += 3
            audio_file = client.files.get(name=audio_file.name)

        if audio_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="音声ファイルの処理に失敗しました。")

        system_instruction = (
            "あなたはプロのコンプライアンス音声監査員です。\n"
            "【最重要指示】音声の全文文字起こし（ベタ貼り）だけを出力することは絶対に禁止します。\n"
            "必ず与えられたプロンプト（監査指示・チェック項目）に従い、音声内容を分析した「総合判定」「スコア」「項目別のOK/NG判定およびタイムスタンプ付き根拠」のみを出力してください。\n\n"
            "【判定上の注意点】\n"
            "* お客様の単なる「言い淀み（どもり）」や「言葉につまづいた状態」だけでマイナス評価にしないでください。\n"
            "* ただし、オペレーター側の高圧的なトーン、強い口調、または話を遮るような話し方の直後に、お客様のトーンが萎縮したり焦ったりした場合は、「オペレーターの応対に起因する顧客の動揺」として厳しく減点・指摘してください。\n"
            "* 単なる言い間違いや迷いと、プレッシャーによる焦りは明確に区別して判定してください。"
        )

        response_text = None
        errors = []

        for model_name in get_target_models():
            success = False
            for attempt in range(6):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[audio_file, prompt],
                        config={"system_instruction": system_instruction},
                    )
                    response_text = response.text
                    success = True
                    print(f"[OK] 使用モデル: {model_name}")
                    break
                except Exception as e:
                    error_text = str(e).upper()
                    # キーが無効なら他のモデルを試しても無駄なので即終了
                    if "API_KEY_INVALID" in error_text or "API KEY NOT VALID" in error_text or "PERMISSION_DENIED" in error_text:
                        raise HTTPException(
                            status_code=500,
                            detail=f"APIキーが無効または権限がありません。Renderの環境変数 GEMINI_API_KEY を確認してください。詳細: {e}",
                        )
                    if any(k in error_text for k in ("503", "UNAVAILABLE", "HIGH DEMAND", "RESOURCE_EXHAUSTED", "429")):
                        wait = min(2 ** attempt, 20)
                        print(f"[Retry {attempt+1}/6] {model_name} busy, retry in {wait}s")
                        time.sleep(wait)
                        continue
                    # 404（モデル廃止）などは次のモデルへ
                    errors.append(f"{model_name}: {e}")
                    print(f"[Skip] {model_name}: {e}")
                    break
            if success:
                break

        if response_text is not None:
            return {"result": response_text}
        raise HTTPException(
            status_code=500,
            detail="解析に失敗しました。詳細: " + (" / ".join(errors) or "全モデルが混雑中でした"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_file is not None:
            try:
                client.files.delete(name=audio_file.name)
            except Exception:
                pass
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
