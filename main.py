import os
import time
import uuid
import secrets
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from google import genai
import uvicorn

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()
security = HTTPBasic()

USERNAME = os.environ.get("AUTH_USER", "admin")
PASSWORD = os.environ.get("AUTH_PASS", "password123")

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

@app.get("/", response_class=HTMLResponse)
async def read_index(username: str = Depends(verify_credentials)):
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html が見つかりません</h1>"

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...), 
    prompt: str = Form(...),
    username: str = Depends(verify_credentials)
):
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".mp3"
    temp_path = f"temp_{uuid.uuid4().hex}{ext}"

    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        audio_file = client.files.upload(file=temp_path)
        
        max_wait = 40
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
        last_error = None

        try:
            target_models = []
            # APIから利用可能なモデルを動的取得
            for m in client.models.list():
                model_name = getattr(m, "name", "")
                methods = getattr(m, "supported_generation_methods", [])
                if "flash" in model_name.lower() and "generateContent" in methods:
                    target_models.append(model_name)

            # 動的取得できなかった場合の保険は現在の最新モデルのみ
            if not target_models:
    target_models = [
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

            # 順番に試行（混雑時はリトライ）
            for model_name in target_models:
                success = False
                for attempt in range(6):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[audio_file, prompt],
                            config={"system_instruction": system_instruction}
                        )
                        response_text = response.text
                        success = True
                        break
                except Exception as e:
                    last_error = e
                    error_text = str(e).upper()

                    if (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                        or "HIGH DEMAND" in error_text
                        or "RESOURCE_EXHAUSTED" in error_text
                    ):
                        wait = min(2 ** attempt, 20)

                        print(f"[Retry {attempt+1}/6] {model_name} 混雑中 {wait}秒後に再試行")

                        time.sleep(wait)
                        continue

                break

            if success:
                break
        except Exception as e:
            last_error = e

        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass
            
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if response_text is not None:
            return {"result": response_text}
        else:
            if last_error:
    error = str(last_error)

    if (
        "503" in error
        or "UNAVAILABLE" in error
        or "HIGH DEMAND" in error.upper()
    ):
        raise HTTPException(
            status_code=503,
            detail="Geminiサーバーが混雑しています。30秒ほど待ってからもう一度お試しください。"
        )

raise HTTPException(
    status_code=500,
    detail=f"解析に失敗しました。詳細: {last_error}"
)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
