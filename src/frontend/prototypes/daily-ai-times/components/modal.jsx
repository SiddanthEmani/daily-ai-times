// Modal for full-article read / video play

function ArticleModal({ story, onClose, savedIds, toggleSave }) {
  const isVideo = story.type === "video";
  const [playing, setPlaying] = useState(isVideo);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const saved = savedIds.has(story.id);
  const body = story.body || [
    story.summary || story.deck || "",
    "The full article continues with expanded reporting, technical detail, and the byline author's contact for tips. This placeholder text stands in for the prose that would appear in the live edition.",
    "Members of the Daily AI Times working press are available at tips@dailyaitimes.example for follow-up questions, corrections, and story ideas from inside the labs.",
  ];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <article className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>× Close</button>
        <div className="kicker">{(story.kicker || story.section).toUpperCase()}</div>
        <h2>{story.headline}</h2>
        <div className="lead-byline" style={{textAlign:'left', margin: '6px 0 16px'}}>
          {story.byline}
          {story.dateline && <><span className="bullet" />{story.dateline}</>}
          <span className="bullet" />Updated {story.time || "today"}
          <span className="bullet" />
          <button className={"save-btn " + (saved ? "saved" : "")} onClick={() => toggleSave(story.id)}>
            {saved ? "★ Saved to clippings" : "☆ Save for later"}
          </button>
        </div>

        {isVideo && (
          <div className="video-surface">
            <PaperImage seed={(story.id || "x").length + 3} label="VIDEO FRAME" />
            {playing ? (
              <>
                <div className="playing-chip">● LIVE PLAYBACK</div>
                <div className="video-bars">
                  {Array.from({ length: 24 }).map((_, i) => (
                    <span key={i} style={{ animationDelay: `${i * 0.07}s` }} />
                  ))}
                </div>
              </>
            ) : (
              <button
                onClick={() => setPlaying(true)}
                style={{
                  position:'absolute', inset: 0, background:'rgba(0,0,0,0.4)', border:'none', cursor:'pointer',
                  color:'#fff', fontFamily:'IBM Plex Mono, monospace', letterSpacing:'0.14em', fontSize: 12
                }}
              >
                ▶ PLAY · {story.duration || "5:00"}
              </button>
            )}
          </div>
        )}

        {!isVideo && (
          <div className="lead-media" style={{marginTop: 8}}>
            <PaperImage seed={(story.id || "x").length + 9} label="ILLUSTRATION" />
          </div>
        )}

        <div className="body">
          {body.map((p, i) => <p key={i}>{p}</p>)}
          <p style={{fontStyle:'italic', color:'var(--ink-soft)'}}>— The Daily AI Times, Desk Edition</p>
        </div>
      </article>
    </div>
  );
}

Object.assign(window, { ArticleModal });
