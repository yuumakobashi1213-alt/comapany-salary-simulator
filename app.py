import json
import re
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types

# ==========================================
# 1. ページ基本設定 ＆ CSS読み込み
# ==========================================
st.set_page_config(
    page_title="日本企業 リアル賃金シミュレーター",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_custom_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

st.markdown('<meta name="robots" content="noindex, nofollow, noarchive">', unsafe_allow_html=True)


# ==========================================
# 2. セキュリティ：合言葉ゲート
# ==========================================
def check_password() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    col_left, col_center, col_right = st.columns([1, 1.5, 1])
    with col_center:
        st.markdown(
            """
            <div class="login-box">
                <h2>🔒 限定公開シミュレーター</h2>
                <p style="color:#64748b; font-size:0.9rem;">
                    リンクと合言葉を知っている方のみご利用いただけます。
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        with st.form("login_form"):
            pwd_input = st.text_input("合言葉を入力", type="password")
            if st.form_submit_button("入室する", use_container_width=True):
                correct_password = st.secrets.get("APP_PASSWORD", "")
                if pwd_input == correct_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("合言葉が違います。")
    return False

if not check_password():
    st.stop()


# ==========================================
# 3. AI連携関数（モデル検知・JSON抽出）
# ==========================================
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key)

def extract_json_from_text(text: str) -> dict:
    if not text:
        raise ValueError("AI応答が空でした。")
    cleaned = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    s_idx, e_idx = cleaned.find("{"), cleaned.rfind("}")
    if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
        cleaned = cleaned[s_idx : e_idx + 1]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return json.loads(cleaned)

def get_available_flash_models(client) -> list:
    preferred = ["gemini-3.8-flash"]
    try:
        for m in client.models.list():
            s_name = getattr(m, "name", "").replace("models/", "")
            if "flash" in s_name.lower() and "2.5" not in s_name and s_name not in preferred:
                preferred.append(s_name)
    except Exception:
        pass
    return preferred

def call_gemini_json(prompt: str) -> dict:
    client = get_gemini_client()
    candidate_models = get_available_flash_models(client)
    for model_name in candidate_models:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            )
            return extract_json_from_text(resp.text)
        except Exception:
            try:
                resp2 = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                return extract_json_from_text(resp2.text)
            except Exception:
                continue
    raise RuntimeError("AIモデルからの応答取得に失敗しました。")


# ==========================================
# 4. AIリサーチ（フェーズ1）
# ==========================================
def fetch_company_jobs(company_name: str) -> dict:
    prompt = f"""
日本の人事制度・労働統計の専門家として「{company_name}」を調査し、JSONのみ出力してください。

【抽出ルール】
1. 企業の公式ドメイン（例：mitsubishicorp.com, ey.com など。ロゴ表示に使用）を特定する。
2. 代表的な募集職種を2〜4つ抽出する。
3. 各職種の「月次初任給」を調べ、「基本給（本給）」と「みなし残業代（固定残業代）」に分解する。みなし残業がない企業は代金0、時間0とする。
4. 家賃補助・独身寮の「若手期の年間経済価値（万円）」と「終了目安年齢（例:34）」を特定する。

【出力JSONスキーマ】
{{
  "company_official_name": "正式な企業名",
  "company_domain": "公式ドメイン",
  "average_annual_income": 1000,
  "average_age": 39.5,
  "jobs": [
    {{
      "job_title": "職種名",
      "init_monthly_base_salary": 25.0,
      "init_monthly_fixed_ot_pay": 5.0,
      "fixed_overtime_hours": 30,
      "default_overtime_hours": 45,
      "housing_annual_value": 100,
      "housing_cutoff_age": 34
    }}
  ]
}}
"""
    return call_gemini_json(prompt)


# ==========================================
# 5. AI推論（フェーズ2）
# ==========================================
def generate_career_simulation(c_data, s_job, eval_tier) -> dict:
    eval_map = {"high": "上位（最速抜擢）", "middle": "平均（標準）", "low": "下位（遅延）"}
    eval_desc = eval_map[eval_tier]

    prompt = f"""
日本の賃金設計専門家として「役職昇進タイムライン」をJSONのみで出力してください。
- 企業：{c_data["company_official_name"]} / 職種：{s_job["job_title"]} / 評価：{eval_desc}

【ルール】
1. その企業の実際の役職ラダー（商社ならS3→S2→M級、コンサルならC→SC→M→SM等）を反映する。
2. 入社(22)、各役職の昇進時、昇進3年後、50歳到達時のチェックポイントを出力。
3. base_salary_real には「基本給＋みなし残業代の合計（年額）」を出力する（超過残業代は含めない）。
4. 管理職（マネージャー・課長級以上）では必ず "is_manager": true にする。

【JSONスキーマ】
{{
  "overview_text": "【概要】全体の解説（3〜4文）",
  "turning_points": [
    {{"badge": "転換点ラベル", "is_manager_jump": false, "title": "見出し", "description": "解説"}}
  ],
  "checkpoints": [
    {{"age": 22, "event_tag": "【入社】", "rank_title": "役職名", "is_manager": false, "base_salary_real": 408, "bonus_real": 87, "overtime_cap_hours": 45}}
  ]
}}
"""
    return call_gemini_json(prompt)


# ==========================================
# 6. 計算エンジン（みなし残業分離対応）
# ==========================================
def calculate_compensation_table(sim_data, s_job, u_base, u_fixed_pay, u_fixed_hrs, u_house, u_overtime, infl) -> pd.DataFrame:
    d_base = float(s_job.get("init_monthly_base_salary", 25.0))
    d_fixed = float(s_job.get("init_monthly_fixed_ot_pay", 0.0))
    d_total = d_base + d_fixed
    u_total = u_base + u_fixed_pay
    
    init_scale = (u_total / d_total) if d_total > 0 else 1.0
    fixed_ot_ratio = (u_fixed_pay / u_total) if u_total > 0 else 0.0

    rows = []
    for cp in sim_data.get("checkpoints", []):
        age = int(cp["age"])
        y_elapsed = max(0, age - 22)
        is_mgr = bool(cp.get("is_manager", False))
        macro = (1.0 + (infl / 100.0)) ** y_elapsed

        # AIが出力した（本給＋みなし残業代）の年額
        total_base_real = float(cp.get("base_salary_real", 360)) * init_scale
        ann_total_base = round(total_base_real * macro)

        if is_mgr:
            ann_base = ann_total_base
            ann_fixed_ot = 0
            ann_extra_ot = 0
            h_text = "適用外（管理職）"
            h_val = 0
        else:
            # みなし残業代と純粋な基本給を比率で分離
            ann_fixed_ot = round(ann_total_base * fixed_ot_ratio)
            ann_base = ann_total_base - ann_fixed_ot
            
            ot_cap = float(cp.get("overtime_cap_hours", 45))
            eff_ot = min(float(u_overtime), ot_cap)
            payable_ot = max(0.0, eff_ot - u_fixed_hrs)
            
            # 超過残業代は「純粋な基本給」から時給を計算して算出
            ann_extra_ot = round((ann_base / 1920.0) * 1.30 * payable_ot * 12.0)
            
            cutoff = int(s_job.get("housing_cutoff_age", 35))
            if age >= cutoff:
                h_text = f"適用外（{cutoff}歳制限）"
                h_val = 0
            else:
                h_val = u_house
                h_text = f"適用（+{h_val}万）" if h_val > 0 else "なし"

        b_real = float(cp.get("bonus_real", 100)) * init_scale
        ann_bonus = round(b_real * macro)
        
        total_inc = ann_base + ann_fixed_ot + ann_extra_ot + ann_bonus
        
        rows.append({
            "age": age, "age_event": f"{age}歳{cp.get('event_tag', '')}", "chart_label": f"{age}歳<br>{cp.get('event_tag', '')}",
            "rank_title": cp.get("rank_title", ""), "is_manager": is_mgr,
            "annual_base": ann_base, "annual_fixed_ot": ann_fixed_ot, "annual_extra_ot": ann_extra_ot,
            "annual_bonus": ann_bonus, "total_income": total_inc,
            "housing_status": h_text
        })
    return pd.DataFrame(rows)


# ==========================================
# 7. サイドバーUI（パラメータ手動修正の超強化）
# ==========================================
st.markdown('<div class="app-header"><h1>📊 日本企業 リアル賃金シミュレーター</h1><p>個社の役職・昇進・みなし残業・福利厚生を完全反映</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ シミュレーション条件")
    c_input = st.text_input("① 企業名を入力", value="三菱商事")
    if st.button("🔍 企業データを取得", use_container_width=True, type="primary"):
        with st.spinner("AIが企業の給与体系とロゴをリサーチ中..."):
            st.session_state.c_data = fetch_company_jobs(c_input)
            st.session_state.sim_result = None

    if "c_data" in st.session_state:
        c_data = st.session_state.c_data
        st.success(f"✅ {c_data['company_official_name']} の情報を取得")
        jobs_list = c_data.get("jobs", [])
        job_titles = [j["job_title"] for j in jobs_list]
        s_job = next((j for j in jobs_list if j["job_title"] == st.selectbox("② 職種を選択", job_titles)), jobs_list[0])

        st.caption("※以下のAI算出値は実態に合わせて自由に修正可能です")
        u_base = st.number_input("③ 月次基本給（本給・万円）", value=float(s_job.get("init_monthly_base_salary", 25.0)), step=0.5)
        u_fixed_pay = st.number_input("④ 月次みなし残業代（万円）", value=float(s_job.get("init_monthly_fixed_ot_pay", 0.0)), step=0.5)
        u_fixed_hrs = st.number_input("⑤ みなし残業時間（h）", value=int(s_job.get("fixed_overtime_hours", 0)), step=1)
        u_house = st.number_input("⑥ 年間家賃補助・社宅価値（万円）", value=int(s_job.get("housing_annual_value", 0)), step=10)
        
        eval_choice = st.radio("⑦ 昇進ペース", ["high", "middle", "low"], index=1, format_func=lambda x: {"high":"🟢 上位抜擢", "middle":"🔵 平均標準", "low":"🟠 下位滞留"}[x])

        if st.button("🚀 推移を算出する", use_container_width=True, type="primary"):
            with st.spinner("役職タイムラインを構築中..."):
                st.session_state.sim_result = generate_career_simulation(c_data, s_job, eval_choice)

        st.divider()
        st.subheader("🎛️ リアルタイム調整スライダー")
        min_ot = int(u_fixed_hrs)
        def_ot = max(min_ot, int(s_job.get("default_overtime_hours", 30)))
        
        # みなし残業がある場合、スライダーの最小値をそこに固定する
        u_overtime = st.slider(f"⑧ 月間想定残業時間（h）", min_value=min_ot, max_value=80, value=def_ot, step=5)
        if min_ot > 0:
            st.caption(f"※みなし残業制のため、下限が {min_ot}時間に固定されています")

        infl_mode = st.radio("⑨ 表示基準（インフレ）", ["real", "nominal_1.0"], format_func=lambda x: {"real":"現在価値（0%）", "nominal_1.0":"将来名目値（+1.0%）"}[x])
        infl_rate = 0.0 if infl_mode == "real" else 1.0


# ==========================================
# 8. メイン出力（グラフ・ロゴ・強調テーブル）
# ==========================================
if "sim_result" in st.session_state and st.session_state.sim_result:
    sim = st.session_state.sim_result
    df = calculate_compensation_table(sim, s_job, u_base, u_fixed_pay, u_fixed_hrs, u_house, u_overtime, infl_rate)

    # ロゴ表示の生成（Google favicon API）
    domain = c_data.get("company_domain", "")
    logo_html = f'<img src="https://s2.googleusercontent.com/s2/favicons?domain={domain}&sz=128" style="width: 44px; height: 44px; border-radius: 8px; vertical-align: middle; margin-right: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">' if domain else ""

    st.markdown(
        f"""
        <div class="summary-box">
            <h3 style="margin-top:0; color:#0f172a; display:flex; align-items:center;">
                {logo_html} {c_data['company_official_name']} / {s_job['job_title']}
            </h3>
            <div class="summary-box-title">【概要】</div>
            {sim.get("overview_text", "")}
        </div>
        """, unsafe_allow_html=True
    )

    # グラフ（みなし：緑、超過：青）
    st.subheader("📊 昇進タイムライン別・年収推移グラフ")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="ベース賃金（本給）", x=df["chart_label"], y=df["annual_base"], marker_color="#1e40af"))
    fig.add_trace(go.Bar(name="みなし残業代", x=df["chart_label"], y=df["annual_fixed_ot"], marker_color="#10b981"))
    fig.add_trace(go.Bar(name="超過残業代", x=df["chart_label"], y=df["annual_extra_ot"], marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="ボーナス（賞与）", x=df["chart_label"], y=df["annual_bonus"], marker_color="#f59e0b"))
    
    fig.add_trace(go.Scatter(name="推定年収", x=df["chart_label"], y=df["total_income"], mode="text", text=df["total_income"].apply(lambda v: f"<b>{v:,}万</b>"), textposition="top center", showlegend=False, hoverinfo="skip"))
    
    fig.update_layout(barmode="stack", height=500, margin=dict(t=40, b=40, l=40, r=20), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # テーブル（総合年収を左端へ、赤色強調）
    st.subheader("📋 昇進タイムライン別・詳細内訳テーブル")
    disp_df = pd.DataFrame({
        "推定年収（合計）": df["total_income"].apply(lambda v: f"{v:,} 万円"),
        "年齢・イベント": df["age_event"],
        "到達役職": df["rank_title"],
        "管理職判定": df["is_manager"].apply(lambda m: "管理職（残業0）" if m else "非管理職"),
        "ベース賃金（本給）": df["annual_base"].apply(lambda v: f"{v:,} 万円"),
        "みなし残業代": df["annual_fixed_ot"].apply(lambda v: "0 万円" if v == 0 else f"{v:,} 万円"),
        "超過残業代": df["annual_extra_ot"].apply(lambda v: "0 万円" if v == 0 else f"{v:,} 万円"),
        "ボーナス": df["annual_bonus"].apply(lambda v: f"{v:,} 万円"),
        "福利厚生（家賃補助等）": df["housing_status"],
    })

    # 合算年収の列だけ薄い赤色背景＋赤太字にするスタイル適用
    styled_df = disp_df.style.apply(
        lambda col: ['background-color: #fee2e2; color: #991b1b; font-weight: 800' if col.name == '推定年収（合計）' else '' for _ in col],
        axis=0
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
