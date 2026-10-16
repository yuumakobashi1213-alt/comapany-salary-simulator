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

# 検索エンジン避け（noindexタグの埋め込み）
st.markdown(
    '<meta name="robots" content="noindex, nofollow, noarchive">',
    unsafe_allow_html=True,
)


# ==========================================
# 2. セキュリティ：合言葉（パスワード）ゲート
# ==========================================
def check_password() -> bool:
    """合言葉を知っている人のみを通す認証ゲート"""
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
                    このサイトはリンクと合言葉を知っている方のみご利用いただけます。
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            pwd_input = st.text_input("合言葉（パスワード）を入力", type="password")
            submitted = st.form_submit_button("入室する", use_container_width=True)
            if submitted:
                correct_password = st.secrets.get("APP_PASSWORD", "")
                if not correct_password:
                    st.error("管理者エラー：Secretsに APP_PASSWORD が設定されていません。")
                elif pwd_input == correct_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("合言葉が違います。もう一度確認してください。")
    return False


if not check_password():
    st.stop()


# ==========================================
# 3. 無料版 Gemini API クライアント ＆ JSON抽出関数
# ==========================================
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Secretsに GEMINI_API_KEY が設定されていません。手順書に従って設定してください。")
        st.stop()
    return genai.Client(api_key=api_key)


def extract_json_from_text(text: str) -> dict:
    """AIの応答テキストから安全にJSONオブジェクトを抽出する"""
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(cleaned)


def call_gemini_json(prompt: str) -> dict:
    """
    無料枠で使える最新の gemini-3.8-flash 等を優先順で呼び出す。
    Google検索グラウンディングを優先試行し、環境制限時は標準JSONモードへ自動フォールバックする。
    """
    client = get_gemini_client()
    candidate_models = [
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ]

    last_error = None
    for model_name in candidate_models:
        # 1st Try: Google検索グラウンディング付きで最新の実態を調査
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2,
                ),
            )
            return extract_json_from_text(response.text)
        except Exception:
            pass

        # 2nd Try: 標準JSON構造化モードで確実に返す
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return extract_json_from_text(response.text)
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"AIモデルからの応答取得に失敗しました。詳細: {last_error}")


# ==========================================
# 4. AIリサーチ関数（フェーズ1：企業＆募集職種取得）
# ==========================================
def fetch_company_jobs(company_name: str) -> dict:
    prompt = f"""
あなたは日本の人事・報酬制度および四季報・有価証券報告書・OpenWork等の労働統計に精通した専門アナリストです。
対象企業：「{company_name}」について調査し、以下のJSON形式のみで出力してください（前後の挨拶や説明文は一切不要）。

【調査・抽出ルール】
1. その企業に実在する新卒・中途の代表的な募集職種（コース）を2〜4つ抽出すること（例：総合職、技術職、ビジネスコンサルタント職、一般職・バックオフィス職など）。
2. 各職種の「月次初任給（万円）※年俸制の場合は年俸÷12の月額換算」を正確に設定すること。
3. 各職種の非管理職（20代〜30代前半）における「実態の月間平均残業時間（時間）」と、「基本給にあらかじめ含まれる固定残業（みなし残業）時間（時間：ない場合は0）」を特定すること。
4. その企業の「家賃補助・独身寮・借上社宅の実態」を調べ、若手〜中堅期の年間経済価値（万円：社宅が手厚い商社・メーカーなら80〜140万、住宅手当なら30〜60万、制度なし・年俸制ファームなら0）と、制度が終了する目安年齢（管理職昇格時や年齢制限：例 34〜35歳）を特定すること。

【出力JSONスキーマ】
{{
  "company_official_name": "正式な企業名",
  "average_annual_income": 1000,
  "average_age": 39.5,
  "housing_policy_summary": "社宅・家賃補助制度の実態に関する簡潔な説明（1〜2文）",
  "jobs": [
    {{
      "job_id": "job_1",
      "job_title": "募集職種名（例：総合職 / ビジネスコンサルタント職など）",
      "init_monthly_salary": 30.0,
      "default_overtime_hours": 35,
      "fixed_overtime_hours": 0,
      "housing_annual_value": 100,
      "housing_cutoff_age": 34,
      "bonus_share_standard": 35,
      "job_note": "初任給やみなし残業・賞与特徴の補足（1文）"
    }}
  ]
}}
"""
    return call_gemini_json(prompt)


# ==========================================
# 5. AI推論関数（フェーズ2：昇進＋3年後タイムライン生成）
# ==========================================
def generate_career_simulation(
    company_data: dict,
    selected_job: dict,
    evaluation_tier: str,
) -> dict:
    eval_label_map = {
        "high": "上位評価（最速抜擢・上位幹部/パートナー/部長到達コース）",
        "middle": "平均評価（標準昇進・標準管理職/課長/室長/シニアマネージャー到達コース）",
        "low": "下位評価（昇進遅延・非管理職長期滞留〜下位管理職頭打ちコース）",
    }
    eval_desc = eval_label_map[evaluation_tier]

    prompt = f"""
あなたは日本企業の人事制度・グレード別賃金テーブルの設計専門家です。
以下の企業・職種・個人評価に基づき、22歳（新卒入社）から50歳までの**「役職昇進イベント駆動型タイムライン」**と**「解説レポート」**をJSON形式のみで出力してください。

【対象条件】
- 企業名：{company_data["company_official_name"]}（有報平均年収目安：{company_data["average_annual_income"]}万円 / 平均年齢：{company_data["average_age"]}歳）
- 採用職種：{selected_job["job_title"]}
- 基準月次初任給：{selected_job["init_monthly_salary"]}万円（初年度年間ベース：{round(selected_job["init_monthly_salary"] * 12)}万円）
- みなし残業（固定残業）時間：月{selected_job["fixed_overtime_hours"]}時間
- 住宅補助・社宅の年間換算価値：{selected_job["housing_annual_value"]}万円（{selected_job["housing_cutoff_age"]}歳または管理職昇格まで適用）
- 入社後の想定評価：{eval_desc}

【タイムライン構築の厳守ルール（最重要）】
1. 固定の5年刻みではなく、この企業の実際の役職ラダー（例：商社ならS3→S2→S1→Mグレード→室長→部長、コンサルならアナリスト/C→SC→M→SM→パートナー、メーカーなら担当→主任→係長→課長→次長→部長）を反映すること。
2. 選択された「想定評価（{eval_desc}）」に合わせて、**何歳でどの役職に昇進するか（昇進スピードと到達上限キャップ）**をリアルに推定すること。
3. 出力するチェックポイント（行）は以下のルールで構成すること：
   - ① 22歳【入社】（初年度）
   - ② 各役職への【昇進】時（その役職の下限スタート基本給＆賞与）
   - ③ 各役職の［昇進の3年後］（同一役職内での習熟号俸アップ＆賞与フル化。※ただし3年以内に次の役職へ昇進する場合はスキップして次の【昇進】を出す）
   - ④ 50歳【到達】（最終到達役職でのキャップ・安定状態）
   ※合計で8〜11個のチェックポイント（年齢昇順）にすること。
4. 各チェックポイントの数値は**「現在価値ベース（インフレ率0%、残業代を除いた基本給と賞与）」**で出力すること（残業代とインフレ底上げはPythonエンジン側で精密計算するため、`base_salary_real` には追加残業代を混ぜないこと。ただしみなし残業制の企業で基本給に固定残業代が内包されている場合はその基本給年額とする）。
5. 管理職（労働基準法上の管理監督者：課長、Mグレード、マネージャー以上）の役職では必ず `"is_manager": true` にし、それ未満の非管理職では `"is_manager": false` にすること。

【出力JSONスキーマ】
{{
  "overview_text": "【概要】として表示するキャリア全体の総括テキスト（3〜4文。初任給からの推移、管理職昇格年齢、50歳到達年収、その会社のベース・賞与・残業代・福利厚生の構造的特徴を具体的に解説）",
  "turning_points": [
    {{
      "age": 25,
      "badge": "昇進イベント種別（例：若手昇格 / 非管理職ピーク / ★管理職ジャンプ / 到達上限など）",
      "is_manager_jump": false,
      "title": "転換点の見出し（例：25歳【S2昇格】／ 賞与のフル支給化と一人前立ち上がり）",
      "description": "そのタイミングで給与構造（ベース・残業代・賞与・福利厚生）がどう変化するかの具体的な解説（2〜3文）"
    }}
  ],
  "checkpoints": [
    {{
      "age": 22,
      "event_tag": "【入社】",
      "rank_title": "個社固有の役職・グレード名（例：G1：担当(S3) / C：コンサルタント）",
      "is_manager": false,
      "base_salary_real": 408,
      "bonus_real": 87,
      "overtime_cap_hours": 40,
      "housing_status_text": "社宅・補助の適用状況テキスト（例：社宅適用（＋120万相当） / ※社宅適用外へ移行 / なし）",
      "housing_value": 120
    }}
  ]
}}
"""
    return call_gemini_json(prompt)


# ==========================================
# 6. Python側 3階建て報酬計算エンジン（リアルタイム再計算）
# ==========================================
def calculate_compensation_table(
    sim_data: dict,
    selected_job: dict,
    user_init_monthly: float,
    user_overtime_hours: int,
    inflation_rate_pct: float,
) -> pd.DataFrame:
    """
    AIが生成した現在価値の役職タイムラインに対し、
    ユーザーの「初任給微調整」「想定残業時間」「インフレ底上げ率」をリアルタイムに反映する。
    """
    default_init = float(selected_job.get("init_monthly_salary", 30.0))
    init_scale = (user_init_monthly / default_init) if default_init > 0 else 1.0
    fixed_ot = float(selected_job.get("fixed_overtime_hours", 0))

    rows = []
    for cp in sim_data.get("checkpoints", []):
        age = int(cp["age"])
        years_elapsed = max(0, age - 22)
        is_manager = bool(cp.get("is_manager", False))

        # ① 外部要因（インフレ・ベア底上げ係数）
        macro_factor = (1.0 + (inflation_rate_pct / 100.0)) ** years_elapsed

        # ② ベース賃金（年間基本給）
        base_real = float(cp.get("base_salary_real", 360)) * init_scale
        annual_base = round(base_real * macro_factor)

        # ③ 残業代（管理職なら強制0円、非管理職なら年代キャップ＆みなし控除後に時給換算）
        if is_manager:
            annual_overtime = 0
        else:
            ot_cap = float(cp.get("overtime_cap_hours", 45))
            effective_ot = min(float(user_overtime_hours), ot_cap)
            payable_ot = max(0.0, effective_ot - fixed_ot)
            # 年間所定労働時間 1,920時間、実効割増率 1.30倍
            annual_overtime = round((annual_base / 1920.0) * 1.30 * payable_ot * 12.0)

        # ④ ボーナス（年間賞与）
        bonus_real = float(cp.get("bonus_real", 100)) * init_scale
        annual_bonus = round(bonus_real * macro_factor)

        # ⑤ 合算年収
        total_income = annual_base + annual_overtime + annual_bonus
        bonus_ratio_pct = round((annual_bonus / total_income) * 100) if total_income > 0 else 0

        rows.append(
            {
                "age": age,
                "age_event": f"{age}歳{cp.get('event_tag', '')}",
                "chart_label": f"{age}歳<br>{cp.get('event_tag', '')}",
                "rank_title": cp.get("rank_title", ""),
                "is_manager": is_manager,
                "annual_base": annual_base,
                "annual_overtime": annual_overtime,
                "annual_bonus": annual_bonus,
                "total_income": total_income,
                "bonus_ratio_pct": f"{bonus_ratio_pct}%",
                "housing_status": cp.get("housing_status_text", "-"),
                "housing_value": int(cp.get("housing_value", 0)),
            }
        )

    return pd.DataFrame(rows)


# ==========================================
# 7. メイン画面レイアウト（ヘッダー ＆ サイドバー入力）
# ==========================================
st.markdown(
    """
    <div class="app-header">
        <h1>📊 日本企業 リアル賃金シミュレーター</h1>
        <p>個社の募集職種・役職テーブル・昇進スピード・管理職での残業代消滅・福利厚生を完全反映</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ シミュレーション条件")

    # Step 1: 企業名検索
    company_input = st.text_input(
        "① 企業名を入力",
        value="三菱商事",
        placeholder="例：三菱商事、EYSC、トヨタ自動車",
    )
    if st.button("🔍 企業の募集職種を調べる", use_container_width=True, type="primary"):
        with st.spinner(f"「{company_input}」の採用職種・初任給・残業・福利厚生を調査中..."):
            try:
                st.session_state.company_data = fetch_company_jobs(company_input)
                st.session_state.sim_result = None
            except Exception as e:
                st.error(f"企業情報の取得中にエラーが発生しました: {e}")

    # 企業データが取得済みの場合、Step 2の条件選択を表示
    if "company_data" in st.session_state and st.session_state.company_data:
        c_data = st.session_state.company_data
        st.success(f"✅ {c_data['company_official_name']} の職種データを取得済み")
        st.caption(f"🏠 **福利厚生実態**: {c_data.get('housing_policy_summary', '')}")

        jobs_list = c_data.get("jobs", [])
        job_titles = [j["job_title"] for j in jobs_list]
        selected_job_title = st.selectbox("② 採用職種を選択", job_titles)
        selected_job = next(
            (j for j in jobs_list if j["job_title"] == selected_job_title), jobs_list[0]
        )

        if selected_job.get("job_note"):
            st.caption(f"💡 {selected_job['job_note']}")

        user_init_monthly = st.number_input(
            "③ 月次初任給（万円・年俸制は÷12換算）",
            min_value=15.0,
            max_value=100.0,
            value=float(selected_job.get("init_monthly_salary", 30.0)),
            step=0.5,
        )

        eval_choice = st.radio(
            "④ 入社後の想定評価（昇進ペース）",
            options=["high", "middle", "low"],
            index=1,
            format_func=lambda x: {
                "high": "🟢 上位（最速抜擢・幹部到達）",
                "middle": "🔵 平均（標準昇進・標準キャップ）",
                "low": "🟠 下位（昇進遅延・滞留）",
            }[x],
        )

        if st.button("🚀 キャリア＆賃金推移を算出する", use_container_width=True, type="primary"):
            with st.spinner("個社の役職テーブルと昇進タイムラインを構築中..."):
                try:
                    st.session_state.sim_result = generate_career_simulation(
                        c_data, selected_job, eval_choice
                    )
                    st.session_state.active_job = selected_job
                    st.session_state.active_eval = eval_choice
                except Exception as e:
                    st.error(f"シミュレーション生成中にエラーが発生しました: {e}")

        st.divider()
        st.subheader("🎛️ リアルタイム調整スライダー")
        st.caption("※以下の設定はAI再検索なしで即座にグラフと表へ反映されます")

        default_ot = int(selected_job.get("default_overtime_hours", 35))
        fixed_ot = int(selected_job.get("fixed_overtime_hours", 0))
        user_overtime = st.slider(
            f"⑤ 月間想定残業時間（企業平均: {default_ot}h / みなし: {fixed_ot}h）",
            min_value=0,
            max_value=70,
            value=default_ot,
            step=5,
        )

        value_mode = st.radio(
            "⑥ 表示基準（インフレ・ベア底上げ）",
            options=["real", "nominal_1.0", "nominal_1.5"],
            index=0,
            format_func=lambda x: {
                "real": "現在価値ベース（インフレ 0.0%）",
                "nominal_1.0": "将来名目値（年率 +1.0% 底上げ込）",
                "nominal_1.5": "将来名目値（年率 +1.5% 底上げ込）",
            }[x],
        )
        inflation_map = {"real": 0.0, "nominal_1.0": 1.0, "nominal_1.5": 1.5}
        inflation_rate = inflation_map[value_mode]


# ==========================================
# 8. メイン画面出力（3セクション構成）
# ==========================================
if "sim_result" not in st.session_state or not st.session_state.sim_result:
    st.info(
        "👈 左側のサイドバーから **「① 企業名」** を入力して職種を取得し、**「🚀 キャリア＆賃金推移を算出する」** を押してください。"
    )
else:
    sim_res = st.session_state.sim_result
    active_job = st.session_state.active_job
    c_name = st.session_state.company_data["company_official_name"]

    df_calc = calculate_compensation_table(
        sim_data=sim_res,
        selected_job=active_job,
        user_init_monthly=user_init_monthly,
        user_overtime_hours=user_overtime,
        inflation_rate_pct=inflation_rate,
    )

    # --- トップKPIサマリーカード（3枚） ---
    first_row = df_calc.iloc[0]
    mgr_rows = df_calc[df_calc["is_manager"] == True]
    mgr_row = mgr_rows.iloc[0] if not mgr_rows.empty else df_calc.iloc[len(df_calc) // 2]
    last_row = df_calc.iloc[-1]

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🌱 初年度（{first_row['age']}歳・{first_row['rank_title']}）</div>
                <div class="kpi-value">{first_row['total_income']:,} 万円</div>
                <div class="kpi-sub">ベース {first_row['annual_base']:,}万 / 残業 {first_row['annual_overtime']:,}万 / 賞与 {first_row['annual_bonus']:,}万</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🚀 管理職到達・中核期（{mgr_row['age']}歳・{mgr_row['rank_title']}）</div>
                <div class="kpi-value">{mgr_row['total_income']:,} 万円</div>
                <div class="kpi-sub">ベース {mgr_row['annual_base']:,}万 / 残業 {mgr_row['annual_overtime']:,}万 / 賞与 {mgr_row['annual_bonus']:,}万</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🏁 50歳到達時点（{last_row['rank_title']}）</div>
                <div class="kpi-value">{last_row['total_income']:,} 万円</div>
                <div class="kpi-sub">ベース {last_row['annual_base']:,}万 / 残業 {last_row['annual_overtime']:,}万 / 賞与 {last_row['annual_bonus']:,}万</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================================================
    # 【エリア 1】文面レポート（概要 ＆ 主要転換点の解説）
    # ==========================================================
    st.subheader(f"📝 1. キャリア＆報酬構造レポート（{c_name} / {active_job['job_title']}）")

    st.markdown(
        f"""
        <div class="summary-box">
            <div class="summary-box-title">【概要】</div>
            {sim_res.get("overview_text", "")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 💡 キャリアの主要転換点（昇進・管理職ジャンプの解説）")
    for tp in sim_res.get("turning_points", []):
        badge_class = "tp-badge-manager" if tp.get("is_manager_jump") else "tp-badge"
        st.markdown(
            f"""
            <div class="turning-point-card">
                <div class="tp-header">
                    <span class="{badge_class}">{tp.get("badge", "転換点")}</span>
                    <span class="tp-title">{tp.get("title", "")}</span>
                </div>
                <p class="tp-desc">{tp.get("description", "")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ==========================================================
    # 【エリア 2】3階建て積み上げグラフ（Plotlyインタラクティブ）
    # ==========================================================
    mode_label = (
        "現在価値ベース（インフレ0%）"
        if inflation_rate == 0
        else f"将来名目値（年+{inflation_rate}%底上げ込）"
    )
    st.subheader(f"📊 2. 昇進タイムライン別・3階建て年収推移グラフ（{mode_label}）")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="ベース賃金（基本給）",
            x=df_calc["chart_label"],
            y=df_calc["annual_base"],
            marker_color="#1e40af",
            text=df_calc["annual_base"].apply(lambda v: f"{v:,}万"),
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>ベース賃金: %{y:,}万円<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="残業代（管理職で消滅）",
            x=df_calc["chart_label"],
            y=df_calc["annual_overtime"],
            marker_color="#10b981",
            text=df_calc["annual_overtime"].apply(lambda v: f"{v:,}万" if v > 0 else ""),
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>残業代: %{y:,}万円<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="ボーナス（賞与）",
            x=df_calc["chart_label"],
            y=df_calc["annual_bonus"],
            marker_color="#f59e0b",
            text=df_calc["annual_bonus"].apply(lambda v: f"{v:,}万" if v > 0 else ""),
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>ボーナス: %{y:,}万円<extra></extra>",
        )
    )

    # 合計年収ラベルを上部に表示
    fig.add_trace(
        go.Scatter(
            name="推定年収（合計）",
            x=df_calc["chart_label"],
            y=df_calc["total_income"],
            mode="text",
            text=df_calc["total_income"].apply(lambda v: f"<b>{v:,}万円</b>"),
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    max_y = int(df_calc["total_income"].max() * 1.15)
    fig.update_layout(
        barmode="stack",
        height=500,
        margin=dict(t=40, b=40, l=40, r=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis=dict(title="金額（万円）", range=[0, max_y], gridcolor="#f1f5f9"),
        xaxis=dict(title="昇進イベント ＆ 3年後タイムライン"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ==========================================================
    # 【エリア 3】昇進タイムライン推移表（データテーブル）
    # ==========================================================
    st.subheader(f"📋 3. 昇進タイムライン別・推定年収内訳テーブル（{mode_label}）")

    display_df = pd.DataFrame(
        {
            "年齢・イベント": df_calc["age_event"],
            "到達役職（グレード）": df_calc["rank_title"],
            "管理職判定": df_calc["is_manager"].apply(
                lambda m: "管理職（残業0）" if m else "非管理職"
            ),
            "ベース賃金": df_calc["annual_base"].apply(lambda v: f"{v:,} 万円"),
            "残業代": df_calc["annual_overtime"].apply(
                lambda v: "0 万円（消滅）" if v == 0 else f"{v:,} 万円"
            ),
            "ボーナス": df_calc["annual_bonus"].apply(lambda v: f"{v:,} 万円"),
            "推定年収（合計）": df_calc["total_income"].apply(lambda v: f"{v:,} 万円"),
            "賞与比率": df_calc["bonus_ratio_pct"],
            "福利厚生（家賃補助・社宅）": df_calc["housing_status"],
        }
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)
