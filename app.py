import streamlit as st
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types

# ページ設定
st.set_page_config(page_title="Speech2face", page_icon="icon.png", layout="centered")
st.title("🎤 Speech2face: VTuberの裏の顔を生成！")

# Gemini APIクライアント
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# アップロード
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください（mp3）(⚠️意図しない画像が生成される場合があります。)", type=["mp3"])
if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("画像を最大4枚生成中..."):
        # 一時ファイル保存
        temp_file_path = "temp_input.mp3"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Gemini用に音声ファイルをアップロード
        try:
            myfile = client.files.upload(file=temp_file_path)
        except Exception as e:
            st.error(f"❌ ファイルアップロードに失敗しました: {e}")
            st.stop()

        # プロンプト定義
        prompt = (
"""
不適切または性的な描写（セクシーさを強調する要素など）は一切含めないでください。
音声から推定される話者の人物像（年齢層、性格的特徴、顔立ちや雰囲気など）をもとに、その人物をリアルで自然な 写真風ポートレート として表現してください。
•	表現スタイル
•	実写に極めて近い写真品質
•	高解像度（4K相当）
•	自然光またはスタジオ光のバランスを意識
•	彩度・コントラストは自然で落ち着いた印象
•	美しく整った仕上がりだが、不自然に加工された印象は避ける
•	構図と内容
•	顔のみを明確に描写（肩から上の範囲を推奨）
•	視線や表情は自然で、人物の性格を反映するもの（例：落ち着いた／親しみやすい／知的な雰囲気など）
•	背景はシンプルかつ自然で、人物を引き立てるもの（例：ぼかしのある単色背景、自然光を感じるニュートラルな空間）
•	禁止事項
•	一切のテキスト（文字、ウォーターマーク、説明文、キャプションなど）を含めない
•	不自然なポーズ、過度な加工、過度に演出された演技的表情は禁止
•	性的・不適切な要素は完全に排除
•	最終的な出力イメージ
•	まるで実際にカメラで撮影されたかのようなリアルな人物写真
•	クリアで高精細な顔立ち
•	自然な肌質・髪の質感
•	人物像の特徴（年齢、性格、雰囲気）を正確に反映
""")

        # 結果格納用リスト
        generated_images = []
        error_messages = []

        for i in range(4):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=[prompt, myfile],
                    config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
                )

                found_image = False
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image = Image.open(BytesIO(part.inline_data.data))
                        generated_images.append(image)
                        found_image = True
                        break  # 画像があったらそれでOK

                if not found_image:
                    msg = "⚠️ テキストのみが返されました（画像なし）"
                    error_messages.append(msg)

            except Exception as e:
                error_messages.append(f"❌ 生成失敗: {str(e)}")

        # 生成された画像の表示
        if generated_images:
            st.subheader("🖼️ 生成された人物画像（最大4枚）")
            cols = st.columns(2)
            for idx, img in enumerate(generated_images):
                with cols[idx % 2]:
                    st.image(img, caption=f"画像 {idx+1}", use_container_width=True)
                    # ダウンロード用にバイナリに変換
                    img_bytes = BytesIO()
                    img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)

                    # ダウンロードボタン
                    st.download_button(
                        label=f"📥 画像 {idx+1} をダウンロード",
                        data=img_bytes,
                        file_name=f"speech2face_{idx+1}.png",
                        mime="image/png"
                    )
        else:
            st.warning("画像は1枚も生成されませんでした。")

        # エラーの表示
        if error_messages:
            st.subheader("📄 生成に失敗した試行のメッセージ")
            for idx, msg in enumerate(error_messages):
                st.markdown(f"**{idx+1}回目:** {msg}")

        # 後処理
        try:
            client.files.delete(name=myfile.name)
        except Exception:
            pass