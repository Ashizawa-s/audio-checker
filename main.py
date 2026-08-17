import os
import time
import uuid
import secrets
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from google import genai
import uvicorn

# --- API キー ---
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()
security = HTTPBasic()

# --- Basic認証の設定（必要に応じてIDとパスワードを変更してください） ---
# デフォルトID: admin / パスワード: password123
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
        
        # 音声ファイルをアップロード
        audio_file = client.files.upload(file=temp_path)
        
        # 処理完了まで待機
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = client.files.get(name=audio_file.name)
            
        if audio_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="音声ファイルの処理に失敗しました。")

        # 利用可能モデルの自動取得
        available_models = []
        try:
            for m in client.models.list():
                model_id = m.name.replace("models/", "")
                if "flash" in model_id or "pro" in model_id:
                    available_models.append(model_id)
        except Exception:
            pass

        if not available_models:
            available_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]

        system_instruction = (
            "あなたはプロのコンプライアンス音声監査員です。\n"
            "【最重要指示】音声の全文文字起こし（ベタ貼り）だけを出力することは絶対に禁止します。\n"
            "必ず与えられたプロンプト（監査指示・チェック項目）に従い、音声内容を分析した「総合判定」「スコア」「項目別のOK/NG判定およびタイムスタンプ付き根拠」のみを出力してください。\n\n"
            "【判定上の注意点】\n"
            "* お客様の単なる「言い淀み（どもり）」や「言葉につまづいた状態」だけでマイナス評価にしないでください。\n"
            "* ただし、オペレーター側の高圧的なトーン、強い口調、または話を遮るような話し方の直後に、お客様のトーンが萎縮したり焦ったりした場合は、「オペレーターの応対に起因する顧客の動揺」として厳しく減点・指摘してください。\n"
            "* 単なる言い間違いや迷いと、プレッシャーによる焦りは明確に区別して判定してください。"
        )

        last_error = None
        response_text = None

        for model_name in available_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[audio_file, prompt],
                    config={"system_instruction": system_instruction}
                )
                response_text = response.text
                break
            except Exception as e:
                last_error = e
                continue

        # 後始末
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass
            
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if response_text is not None:
            return {"result": response_text}
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"利用可能なモデルでの解析に失敗しました。詳細: {last_error}"
            )

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
