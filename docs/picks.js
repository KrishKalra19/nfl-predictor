// Website picks: sign in, file a probability per game before kickoff, get graded against the model.
// Storage and the kickoff lock live in Supabase (see supabase/schema.sql); its access rules refuse
// late filings and keep everyone's picks private until the game kicks off. This key is public by design.
(() => {
  const SB_URL = "https://weuhagxpoxlhctuxpbaq.supabase.co";
  const SB_KEY = "sb_publishable_FGIbLSQ-cKbuJgJokoBrUw_c6UCwrug";
  if (!window.supabase || typeof DATA === "undefined") return;
  const sb = window.supabase.createClient(SB_URL, SB_KEY);
  const bar = document.getElementById("filings");
  const games = DATA.this_week.filter((g) => g.id);
  const started = (g) => g.kickoff && new Date(g.kickoff) <= new Date();
  const mk = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const pc = (p) => Math.round(p * 100) + "%";
  let mine = {};          // game_id -> p_home already filed
  const unsealed = new Set();
  bar.hidden = false;

  // ---------- sign in / join
  function authForm(mode, msg) {
    bar.replaceChildren();
    const f = mk("form", "auth");
    const head = mk("div", "auth-head");
    head.append(mk("span", "lbl-sm", "File your own picks"),
      mk("span", "small muted", mode === "join"
        ? "Join the firm: pick a name for the leaderboard, plus an email and password to sign back in."
        : "Sign in to file a pick on each game. The model's ruling stays sealed until you file."));
    const name = mk("input"); Object.assign(name, { placeholder: "Name on the leaderboard", maxLength: 30, required: true, autocomplete: "nickname" });
    const email = mk("input"); Object.assign(email, { type: "email", placeholder: "Email", required: true, autocomplete: "email" });
    const pw = mk("input"); Object.assign(pw, { type: "password", placeholder: "Password (6+ characters)", required: true, minLength: 6,
      autocomplete: mode === "join" ? "new-password" : "current-password" });
    const go = mk("button", "btn", mode === "join" ? "Join" : "Sign in"); go.type = "submit";
    const sw = mk("button", "link", mode === "join" ? "Have an account? Sign in" : "New here? Join the firm"); sw.type = "button";
    sw.onclick = () => authForm(mode === "join" ? "in" : "join");
    const fields = mk("div", "auth-fields");
    if (mode === "join") fields.append(name);
    fields.append(email, pw, go);
    const out = mk("span", "small status", msg || "");
    f.append(head, fields, mk("div", "auth-foot"));
    f.lastChild.append(sw, out);
    f.onsubmit = async (e) => {
      e.preventDefault(); go.disabled = true; out.textContent = "…";
      const r = mode === "join"
        ? await sb.auth.signUp({ email: email.value.trim(), password: pw.value, options: { data: { name: name.value.trim() } } })
        : await sb.auth.signInWithPassword({ email: email.value.trim(), password: pw.value });
      go.disabled = false;
      if (r.error) { out.textContent = r.error.message; return; }
      if (mode === "join" && !r.data.session) { out.textContent = "Check your email to confirm, then sign in."; return; }
    };
    bar.append(f);
  }

  async function ensureProfile(user) {
    const { data } = await sb.from("profiles").select("name").eq("id", user.id).maybeSingle();
    if (data) return data.name;
    const name = (user.user_metadata && user.user_metadata.name) || user.email.split("@")[0];
    await sb.from("profiles").insert({ id: user.id, name: name.slice(0, 30) });
    return name;
  }

  async function signedIn(user) {
    const name = await ensureProfile(user);
    const ids = games.map((g) => g.id);
    const { data } = ids.length ? await sb.from("picks").select("game_id,p_home").eq("user_id", user.id).in("game_id", ids) : { data: [] };
    mine = Object.fromEntries((data || []).map((r) => [r.game_id, r.p_home]));
    bar.replaceChildren();
    const row = mk("div", "auth-in");
    const who = mk("span"); who.append("Signed in as ", mk("b", null, name));
    const count = mk("span", "small muted"); count.id = "filed-count";
    const out = mk("button", "link", "Sign out"); out.type = "button";
    out.onclick = () => sb.auth.signOut();
    row.append(who, count, out);
    bar.append(row);
    games.forEach((g) => decorate(g, true));
    updateCount();
  }

  function updateCount() {
    const c = document.getElementById("filed-count");
    if (c) c.textContent = `${games.filter((g) => mine[g.id] != null).length} of ${games.length} filed this week`;
  }

  // ---------- one slider per case
  function decorate(g, on) {
    const card = document.querySelector(`.case[data-id="${g.id}"]`);
    if (!card) return;
    card.querySelectorAll(".file, .seal-note").forEach((e) => e.remove());
    card.classList.remove("sealed");
    if (!on) return;
    const locked = started(g);
    const filed = mine[g.id];
    if (!locked && filed == null && !unsealed.has(g.id)) {
      card.classList.add("sealed");
      const n = mk("div", "seal-note small muted");
      n.append("The model's ruling is sealed until you file. ");
      const u = mk("button", "link", "Unseal anyway"); u.type = "button";
      u.onclick = () => { unsealed.add(g.id); decorate(g, true); };
      n.append(u);
      card.querySelector(".caption").after(n);
    }
    const box = mk("div", "file" + (locked ? " locked" : ""));
    const lab = mk("label", "lbl-sm", "Your filing");
    const sl = mk("input"); Object.assign(sl, { type: "range", min: 1, max: 99, step: 1, value: Math.round((filed ?? 0.5) * 100), disabled: locked });
    sl.id = "f-" + g.id; lab.htmlFor = sl.id;
    sl.setAttribute("aria-label", `Chance ${g.home} beats ${g.away}`);
    const ends = mk("div", "file-row");
    ends.append(mk("span", null, g.away), sl, mk("span", null, g.home));
    const read = mk("span", "file-read");
    const btn = mk("button", "btn", filed != null ? "Update" : "File"); btn.type = "button";
    const st = mk("span", "small status");
    const show = () => {
      const p = sl.value / 100;
      read.textContent = p === 0.5 ? "Coin flip" : `${p > 0.5 ? g.home : g.away} ${pc(Math.max(p, 1 - p))}`;
      sl.style.setProperty("--fill", sl.value + "%");
    };
    sl.oninput = () => { show(); st.textContent = ""; };
    show();
    btn.onclick = async () => {
      btn.disabled = true; st.textContent = "Filing…";
      const p = sl.value / 100;
      const { error } = await sb.from("picks").upsert({ game_id: g.id, p_home: p }, { onConflict: "user_id,game_id" });
      btn.disabled = false;
      if (error) { st.textContent = /row-level|policy/i.test(error.message) ? "Locked: this game has kicked off." : error.message; return; }
      const first = mine[g.id] == null;
      mine[g.id] = p;
      st.textContent = "Filed ✓";
      btn.textContent = "Update";
      if (first) { decorate(g, true); document.querySelector(`.case[data-id="${g.id}"] .file .status`).textContent = "Filed ✓"; }
      updateCount();
    };
    const foot = mk("div", "file-foot");
    if (locked) foot.append(read, mk("span", "small muted", filed != null ? "Locked at kickoff" : "Kicked off — no filing"));
    else foot.append(read, btn, st);
    box.append(lab, ends, foot);
    card.querySelector(".caption").after(box);
  }

  // ---------- leaderboard: every user's picks on games that kicked off and have a final
  const ll = (p, y) => { p = Math.min(Math.max(p, 0.01), 0.99); return -(y * Math.log(p) + (1 - y) * Math.log(1 - p)); };
  async function leaderboard() {
    const finals = DATA.scores || {};
    const ids = Object.keys(finals);
    if (!ids.length) return;
    const rows = [];
    for (let i = 0; i < ids.length; i += 150) {   // keep URLs short
      const { data, error } = await sb.from("picks").select("user_id,game_id,p_home").in("game_id", ids.slice(i, i + 150));
      if (error) return;
      rows.push(...data);
    }
    if (!rows.length) return;
    const { data: profs } = await sb.from("profiles").select("id,name");
    const names = Object.fromEntries((profs || []).map((p) => [p.id, p.name]));
    const by = {};
    for (const r of rows) (by[r.user_id] = by[r.user_id] || []).push(r);
    const table = Object.entries(by).map(([uid, rs]) => {
      let won = 0, lost = 0, sum = 0, msum = 0, mn = 0;
      for (const r of rs) {
        const [a, h] = finals[r.game_id];
        const y = h > a ? 1 : h < a ? 0 : 0.5;
        sum += ll(r.p_home, y);
        if (y !== 0.5 && Math.abs(r.p_home - 0.5) > 0.004) ((r.p_home > 0.5) === (y === 1) ? won++ : lost++);
        const m = DATA.model_p && DATA.model_p[r.game_id];
        if (m != null) { msum += ll(m, y) - ll(r.p_home, y); mn++; }
      }
      return { name: names[uid] || "Associate", n: rs.length, won, lost, ll: sum / rs.length, vs: mn ? -msum / mn : null };
    }).sort((a, b) => a.ll - b.ll);
    const body = document.getElementById("assoc-body");
    body.replaceChildren();
    for (const t of table) {
      const tr = mk("tr");
      tr.append(mk("td", null, t.name), mk("td", "mono", String(t.n)), mk("td", "mono", `${t.won}–${t.lost}`),
        mk("td", "mono", t.ll.toFixed(3)), mk("td", "mono", t.vs == null ? "—" : (t.vs > 0 ? "+" : "") + t.vs.toFixed(3)));
      body.append(tr);
    }
    document.getElementById("associates").hidden = false;
  }

  // Supabase calls can't run inside the auth callback itself, so hand off with a timeout
  sb.auth.onAuthStateChange((ev, session) => {
    if (ev === "TOKEN_REFRESHED" || ev === "USER_UPDATED") return;
    setTimeout(() => {
      if (session && session.user) signedIn(session.user);
      else { mine = {}; games.forEach((g) => decorate(g, false)); authForm("in"); }
    }, 0);
  });
  leaderboard();
})();
