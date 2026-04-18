// Above-the-fold: briefing, lead story, swing story

function Briefing({ onOpen, extras = [] }) {
  const items = [
    { kicker: "The Exit", text: "Weil and Peebles to launch $1.8B fund; OpenAI confirms departures." },
    { kicker: "Chips", text: "Nitro isolation layer formally verified end-to-end for training clusters." },
    { kicker: "Policy", text: "EU begins AI Act Phase II; first audits due Q3." },
    { kicker: "Research", text: "CRISPR payload 38% smaller via AI-guided protein design." },
    { kicker: "Markets", text: "Hyperscaler capex guidance raised; semis lead gainers on the day." },
  ];
  return (
    <section>
      <h3 className="briefing-title">This Morning’s Briefing</h3>
      <ol className="briefing-list">
        {items.map((it, i) => (
          <li key={i}>
            <span className="briefing-num">{String(i+1).padStart(2, "0")}</span>
            <div className="briefing-text">
              <span className="kicker">{it.kicker}</span>
              {it.text}
            </div>
          </li>
        ))}
      </ol>

      <h3 className="briefing-title" style={{ marginTop: 28 }}>Also In The News</h3>
      <div className="also-stack">
        {extras.map((s) => (
          <article key={s.id} className="also-item" onClick={() => onOpen(s)}>
            <div className="also-section">{s.section}{s.type === "video" ? " · Video" : ""}</div>
            <h4 className="also-headline">{s.headline}</h4>
            <div className="also-summary">{s.summary}</div>
            <div className="also-meta">
              <span>{s.byline}</span>
              <span>{s.time}</span>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function LeadStory({ story, onOpen, savedIds, toggleSave }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <section>
      <div className="lead-kicker">{story.kicker} · {story.section.toUpperCase()}</div>
      <h1 className="lead-headline" onClick={() => onOpen(story)} style={{cursor:'pointer'}}>
        {story.headline}
      </h1>
      <p className="lead-deck">{story.deck}</p>

      <div className="lead-media" onClick={() => onOpen(story)}>
        <PaperImage seed={1} label="PORTRAIT · STAFF PHOTOGRAPHER" />
      </div>
      <div className="lead-caption">
        {story.hero.caption}
        <div style={{ marginTop: 4, opacity: 0.7 }}>— {story.hero.credit}</div>
      </div>

      <div className="lead-byline">
        {story.byline}
        <span className="bullet" />
        {story.dateline}
        <span className="bullet" />
        Updated {story.time}
      </div>

      <div className="lead-body">
        {(expanded ? story.body : story.body.slice(0, 3)).map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>
      <button className="lead-continued" onClick={() => onOpen(story)}>
        Continue reading  →  Page A2
      </button>
    </section>
  );
}

function SwingStory({ story, onOpen }) {
  return (
    <section>
      <div className="swing-kicker">{story.kicker}</div>
      <h2 className="swing-headline" onClick={() => onOpen(story)} style={{cursor:'pointer'}}>
        {story.headline}
      </h2>
      <div className="swing-byline">{story.byline}</div>
      <div className="swing-body">
        {story.body.slice(0, 4).map((p, i) => <p key={i}>{p}</p>)}
        <button className="lead-continued" onClick={() => onOpen(story)}>
          Continue on Page A4  →
        </button>
      </div>
    </section>
  );
}

Object.assign(window, { Briefing, LeadStory, SwingStory });
