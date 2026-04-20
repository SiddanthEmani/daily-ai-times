// Shared components: Masthead, Ticker, Nav, placeholder media

const { useState, useEffect, useRef } = React;

// ---------- Ticker ----------
function Ticker() {
  const [paused, setPaused] = useState(false);
  const items = window.TICKER_ITEMS;
  // duplicate for seamless loop
  const strip = [...items, ...items];
  return (
    <div className="ticker">
      <div className="ticker-track" onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)}>
        <div className={"ticker-strip " + (paused ? "paused" : "")}>
          {strip.map((it, i) => (
            <span key={i} className="ticker-item">
              <span className="ticker-tag">{it.tag}</span>
              <span>{it.text}</span>
              <span className="ticker-dot" />
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------- Masthead ----------
function Masthead() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 30000);
    return () => clearInterval(t);
  }, []);
  const dateStr = now.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
  const timeStr = now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  return (
    <header className="masthead-wrap">
      <div className="masthead-topbar">
        <div className="left mast-meta-line">
          <span>Vol. XII · No. 1,508</span>
        </div>
        <div className="center">
          <span>San Francisco</span>
        </div>
        <div className="right mast-meta-line" style={{justifyContent:'flex-end'}}>
          <span>{dateStr}</span>
          <span>{timeStr}</span>
        </div>
      </div>

      <h1 className="masthead-title" style={{ fontFamily: '"Playfair Display", serif' }}>
        Daily <span className="amp">AI</span> Times
      </h1>

      <div className="masthead-motto" style={{ lineHeight: 1.3, borderWidth: 0 }}>
        <span style={{ fontWeight: 800 }}>A Newspaper for the Working Engineer</span>
        <span className="dot" />
        <span style={{ fontWeight: 600 }}>Is AGI Already Among Us?</span>
        <span className="dot" />
        <span>Pollution &amp; Energy Index</span>
      </div>
    </header>
  );
}

// ---------- Section Nav ----------
function Nav({ section, setSection, query, setQuery, counts }) {
  const sections = ["All", "Research", "Industry", "Tools", "Policy", "Video", "Opinion"];
  return (
    <nav className="nav" role="tablist">
      {sections.map(s => (
        <button
          key={s}
          className={"nav-btn " + (section === s ? "active" : "")}
          onClick={() => setSection(s)}
          role="tab"
        >
          {s}
          {counts[s] != null && <span className="count">{counts[s]}</span>}
        </button>
      ))}
      <div className="nav-search">
        <span>⌕</span>
        <input
          placeholder="Search the paper"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
      </div>
    </nav>
  );
}

// ---------- Placeholder media (no external images) ----------
// Produces a deterministic newsprint-style illustration for each article.
function PaperImage({ seed = 0, label = "PHOTOGRAPH", tall = false }) {
  // deterministic pseudo-random
  const rand = (i) => {
    const x = Math.sin(seed * 9301 + i * 49297) * 233280;
    return x - Math.floor(x);
  };
  const hue = Math.floor(rand(1) * 40) + 20; // warm range
  const tone1 = `oklch(0.68 0.05 ${hue})`;
  const tone2 = `oklch(0.42 0.04 ${hue + 20})`;
  const tone3 = `oklch(0.82 0.03 ${hue})`;
  const shapes = Math.floor(rand(2) * 3) + 2;
  return (
    <svg viewBox="0 0 400 250" preserveAspectRatio="xMidYMid slice" style={{width:'100%', height:'100%', display:'block'}}>
      <defs>
        <pattern id={`hatch-${seed}`} width="3" height="3" patternUnits="userSpaceOnUse" patternTransform={`rotate(${45 + rand(3)*30})`}>
          <line x1="0" y1="0" x2="0" y2="3" stroke="rgba(0,0,0,0.18)" strokeWidth="1" />
        </pattern>
        <pattern id={`dots-${seed}`} width="4" height="4" patternUnits="userSpaceOnUse">
          <circle cx="2" cy="2" r="0.7" fill="rgba(0,0,0,0.25)" />
        </pattern>
        <linearGradient id={`sky-${seed}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={tone3} />
          <stop offset="100%" stopColor={tone1} />
        </linearGradient>
      </defs>
      <rect width="400" height="250" fill={`url(#sky-${seed})`} />
      <rect width="400" height="250" fill={`url(#dots-${seed})`} />
      {/* horizon / ground */}
      <rect x="0" y={150 + rand(4)*40} width="400" height="250" fill={tone2} opacity="0.45" />
      <rect x="0" y={150 + rand(4)*40} width="400" height="250" fill={`url(#hatch-${seed})`} />
      {/* a subject silhouette — abstract portrait or object */}
      {Array.from({ length: shapes }).map((_, i) => {
        const cx = 80 + i * 90 + rand(10+i) * 30;
        const cy = 160 + rand(20+i) * 20;
        const r = 28 + rand(30+i) * 18;
        return (
          <g key={i}>
            <ellipse cx={cx} cy={cy + r*0.9} rx={r*0.7} ry={r*0.25} fill="rgba(0,0,0,0.2)" />
            <circle cx={cx} cy={cy - r*0.4} r={r*0.38} fill={tone2} opacity="0.85" />
            <rect x={cx - r*0.45} y={cy - r*0.1} width={r*0.9} height={r*0.85} rx="4" fill={tone2} opacity="0.85" />
          </g>
        );
      })}
      <rect x="0" y="0" width="400" height="250" fill="none" stroke="rgba(0,0,0,0.1)" strokeWidth="1" />
      <text x="12" y="240" fontFamily="IBM Plex Mono, monospace" fontSize="9" fill="rgba(0,0,0,0.5)" letterSpacing="1">
        {label}
      </text>
    </svg>
  );
}

Object.assign(window, { Ticker, Masthead, Nav, PaperImage });
