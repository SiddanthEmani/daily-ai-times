// Below-the-fold stories + sidebar boxes

function StoryCard({ story, idx, onOpen, savedIds, toggleSave, focused }) {
  const saved = savedIds.has(story.id);
  const hasMedia = story.type === "video" || story.type === "photo";
  return (
    <article
      className={"story " + (story.type === "video" ? "featured" : "")}
      data-story-id={story.id}
      style={focused ? { outline: "2px solid var(--accent)", outlineOffset: "4px" } : null}
    >
      <div className="story-section">
        <span>{story.section}</span>
        <span className="sep" />
        {story.type === "video" && <span style={{color:'var(--accent)'}}>VIDEO</span>}
        {story.type === "photo" && <span style={{color:'var(--ink-soft)'}}>WITH PHOTO</span>}
      </div>

      {hasMedia && (
        <div className="media-thumb" onClick={() => onOpen(story)}>
          <PaperImage seed={idx + 10} label={story.type === "video" ? "VIDEO STILL" : "PHOTOGRAPH"} />
          {story.type === "video" && (
            <>
              <div className="video-chip">▶ Watch</div>
              <div className="play-overlay">
                <div className="play-icon" />
              </div>
              <div className="duration">{story.duration}</div>
            </>
          )}
        </div>
      )}

      <h3 className="story-headline" onClick={() => onOpen(story)}>{story.headline}</h3>
      <p className="story-summary">{story.summary}</p>
      <div className="story-meta">
        <span className="byline">{story.byline}</span>
        <span style={{display:'flex', gap: 10, alignItems:'center'}}>
          <span>{story.time}</span>
          <button
            className={"save-btn " + (saved ? "saved" : "")}
            onClick={(e) => { e.stopPropagation(); toggleSave(story.id); }}
            title={saved ? "Remove from saved" : "Save for later"}
          >
            {saved ? "★ Saved" : "☆ Save"}
          </button>
        </span>
      </div>
    </article>
  );
}

function MarketsBox() {
  const m = window.ARTICLES.markets;
  return (
    <aside className="box">
      <div className="box-title">
        <span>Markets</span>
        <span className="chip">● LIVE · 11:42 ET</span>
      </div>
      <div className="markets-grid">
        {m.map((r, i) => (
          <div key={i} className="markets-row">
            <span className="t">{r.ticker}</span>
            <span>{r.px}</span>
            <span className={"ch " + (r.up ? "up" : "down")}>{r.ch}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}

function WeatherBox() {
  const w = window.ARTICLES.weather;
  return (
    <aside className="box">
      <div className="box-title">
        <span>Cities</span>
        <span className="chip">° FAHRENHEIT</span>
      </div>
      {w.map((r, i) => (
        <div key={i} className="weather-row">
          <span className="city">{r.city}</span>
          <span className="temps">{r.hi}° / {r.lo}°</span>
          <span className="cond">{r.cond}</span>
        </div>
      ))}
    </aside>
  );
}

function OpinionBox({ onOpen }) {
  const items = window.ARTICLES.opinion;
  return (
    <aside className="box">
      <div className="box-title">
        <span>Opinion</span>
        <span className="chip">EDITORIAL</span>
      </div>
      {items.map((o) => (
        <div key={o.id} className="opinion-item">
          <h4 className="opinion-title" onClick={() => onOpen({
            id: o.id, section: "Opinion", headline: o.title, byline: "By " + o.author.toUpperCase(),
            body: [o.excerpt, "This column continues inside. The full text is available to subscribers and to members of the working press."],
            time: "today", type: "text"
          })}>{o.title}</h4>
          <div className="opinion-author">By {o.author}</div>
          <div className="opinion-excerpt">“{o.excerpt}”</div>
        </div>
      ))}
    </aside>
  );
}

function SavedBox({ savedIds, allStories, onOpen, toggleSave }) {
  if (savedIds.size === 0) return null;
  const saved = allStories.filter(s => savedIds.has(s.id));
  return (
    <aside className="box" style={{borderColor: "var(--accent)"}}>
      <div className="box-title" style={{borderBottomColor:"var(--accent)"}}>
        <span>Your Clippings</span>
        <span className="chip">{savedIds.size} SAVED</span>
      </div>
      {saved.map(s => (
        <div key={s.id} className="opinion-item">
          <h4 className="opinion-title" onClick={() => onOpen(s)}>{s.headline}</h4>
          <div className="opinion-author">
            {s.section}
            <button className="save-btn saved" style={{float:'right'}} onClick={() => toggleSave(s.id)}>REMOVE</button>
          </div>
        </div>
      ))}
    </aside>
  );
}

Object.assign(window, { StoryCard, MarketsBox, WeatherBox, OpinionBox, SavedBox });
