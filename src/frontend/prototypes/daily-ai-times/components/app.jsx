// Main App — wiring together the whole frontpage

function App() {
  const A = window.ARTICLES;
  const [section, setSection] = useState("All");
  const [query, setQuery] = useState("");
  const [openStory, setOpenStory] = useState(null);
  const [savedIds, setSavedIds] = useState(() => {
    try { return new Set(JSON.parse(localStorage.getItem("dat_saved") || "[]")); }
    catch { return new Set(); }
  });
  const [focusIdx, setFocusIdx] = useState(-1);

  // tweaks with EDITMODE persistence
  const defaults = /*EDITMODE-BEGIN*/{
    "density": "comfortable",
    "tone": "cream",
    "photos": "on",
    "ticker": "on"
  }/*EDITMODE-END*/;
  const [tweaks, setTweaks] = useState(defaults);
  const [tweaksVisible, setTweaksVisible] = useState(false);

  // apply tweak attrs
  useEffect(() => {
    document.documentElement.dataset.density = tweaks.density;
    document.documentElement.dataset.tone = tweaks.tone;
    document.documentElement.dataset.photos = tweaks.photos;
  }, [tweaks]);

  // edit-mode host protocol
  useEffect(() => {
    const onMsg = (e) => {
      if (!e?.data?.type) return;
      if (e.data.type === "__activate_edit_mode") setTweaksVisible(true);
      if (e.data.type === "__deactivate_edit_mode") setTweaksVisible(false);
    };
    window.addEventListener("message", onMsg);
    window.parent.postMessage({ type: "__edit_mode_available" }, "*");
    return () => window.removeEventListener("message", onMsg);
  }, []);

  // persist saved clippings
  useEffect(() => {
    localStorage.setItem("dat_saved", JSON.stringify([...savedIds]));
  }, [savedIds]);

  const toggleSave = (id) => {
    setSavedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  // filter stories by section + query
  const allStories = [A.lead, A.swing, ...A.stories];
  const matchesSection = (s) => {
    if (section === "All") return true;
    if (section === "Video") return s.type === "video";
    if (section === "Opinion") return s.section === "Opinion";
    return s.section === section;
  };
  const matchesQuery = (s) => {
    if (!query.trim()) return true;
    const q = query.toLowerCase();
    return [s.headline, s.summary, s.section, s.byline, s.deck].filter(Boolean).join(" ").toLowerCase().includes(q);
  };

  const filteredGrid = A.stories.filter(s => matchesSection(s) && matchesQuery(s));
  const showAbove = section === "All" && !query.trim();

  const counts = {
    All: A.stories.length + 2,
    Research: A.stories.filter(s => s.section === "Research").length,
    Industry: A.stories.filter(s => s.section === "Industry").length + (A.lead.section === "Industry" ? 1 : 0),
    Tools: A.stories.filter(s => s.section === "Tools").length,
    Policy: A.stories.filter(s => s.section === "Policy").length,
    Video: A.stories.filter(s => s.type === "video").length,
    Opinion: A.opinion.length,
  };

  // keyboard nav j/k between grid cards, Enter to open
  useEffect(() => {
    const onKey = (e) => {
      if (openStory) return;
      if (e.target?.tagName === "INPUT") return;
      if (e.key === "j") { e.preventDefault(); setFocusIdx(i => Math.min(filteredGrid.length - 1, i + 1)); }
      else if (e.key === "k") { e.preventDefault(); setFocusIdx(i => Math.max(0, i - 1)); }
      else if (e.key === "Enter" && focusIdx >= 0) { setOpenStory(filteredGrid[focusIdx]); }
      else if (e.key === "Escape") setFocusIdx(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [filteredGrid, focusIdx, openStory]);

  useEffect(() => {
    if (focusIdx >= 0) {
      const el = document.querySelector(`[data-story-id="${filteredGrid[focusIdx]?.id}"]`);
      el?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
    }
  }, [focusIdx]);

  // distribute grid stories into 4 columns round-robin, but put sidebar boxes into col 4
  const cols = [[], [], [], []];
  // reserve a few stories for the "Also In The News" stack under the Briefing on front page
  const briefingExtras = showAbove ? A.stories.slice(0, 3) : [];
  const gridStories = showAbove ? filteredGrid.filter(s => !briefingExtras.includes(s)) : filteredGrid;
  gridStories.forEach((s, i) => cols[i % 4].push({ kind: "story", story: s, idx: i }));

  // sidebar boxes in col 4, spaced out
  if (showAbove) {
    cols[3].splice(1, 0, { kind: "box", node: <MarketsBox key="markets" /> });
    cols[3].splice(3, 0, { kind: "box", node: <WeatherBox key="weather" /> });
    cols[3].push({ kind: "box", node: <OpinionBox key="op" onOpen={setOpenStory} /> });
    if (savedIds.size > 0) cols[0].push({ kind: "box", node: <SavedBox key="saved" savedIds={savedIds} allStories={allStories} onOpen={setOpenStory} toggleSave={toggleSave} /> });

    // "In Other News" tail — compact headline lists at the bottom of cols 0-2
    // so they visually match col 3 which is taller because of its sidebar boxes.
    const tailPools = [
      [
        "Post-training teams are the new status symbol inside frontier labs",
        "A quiet move to smaller, specialist models at three major banks",
        "Retrieval evaluations finally get a standardized benchmark suite",
        "Open letter: senior researchers call for reproducibility covenants",
        "Why inference cost curves are flattening sooner than expected",
      ],
      [
        "Chip startups pitch ‘vertical integration’ to skeptical buyers",
        "A union push at one major lab fizzles, for now",
        "The rise of the internal ‘AI platform’ role, explained",
        "Evaluation teams are hiring; researchers are not. A data note.",
        "How three universities are rewriting their CS curricula",
      ],
      [
        "Edge inference returns as latency SLAs tighten across the stack",
        "Agents meet accounting: what the early deployments are teaching",
        "A survey of formal verification in ML pipelines",
        "Notes from a week inside a model-evaluation contractor",
        "Why the ‘data flywheel’ language is being retired, quietly",
      ],
    ];
    tailPools.forEach((items, ci) => {
      cols[ci].push({ kind: "tail", title: ci === 0 ? "In Other News" : ci === 1 ? "On The Wire" : "From The Desks", items });
    });
  }

  return (
    <>
      {tweaks.ticker === "on" && <Ticker />}
      <div className="page">
        <Masthead />
        <Nav section={section} setSection={setSection} query={query} setQuery={setQuery} counts={counts} />

        {showAbove && (
          <div className="above">
            <Briefing onOpen={setOpenStory} extras={briefingExtras} />
            <div className="vrule" />
            <LeadStory story={A.lead} onOpen={setOpenStory} savedIds={savedIds} toggleSave={toggleSave} />
            <div className="vrule" />
            <SwingStory story={A.swing} onOpen={setOpenStory} />
          </div>
        )}

        {!showAbove && (
          <div style={{padding: '22px 0 10px', borderBottom: '1px solid var(--rule-soft)', fontFamily:'IBM Plex Mono, monospace', fontSize: 12, letterSpacing: '0.14em', textTransform:'uppercase', color:'var(--ink-soft)'}}>
            Showing <strong style={{color:'var(--ink)'}}>{filteredGrid.length}</strong> stories
            {section !== "All" && <> in <strong style={{color:'var(--accent)'}}>{section}</strong></>}
            {query.trim() && <> matching “<strong style={{color:'var(--accent)'}}>{query}</strong>”</>}
          </div>
        )}

        <div className="below">
          {cols.map((col, ci) => (
            <div className="col" key={ci}>
              {col.map((item, i) => {
                if (item.kind === "story") {
                  return (
                    <StoryCard
                      key={item.story.id}
                      story={item.story}
                      idx={item.idx}
                      onOpen={setOpenStory}
                      savedIds={savedIds}
                      toggleSave={toggleSave}
                      focused={focusIdx === item.idx}
                    />
                  );
                }
                if (item.kind === "tail") {
                  return (
                    <section key={i} className="tail-section">
                      <h4 className="tail-title">{item.title}</h4>
                      <ul className="tail-list">
                        {item.items.map((h, j) => (
                          <li key={j} onClick={() => setOpenStory({
                            id: `tail-${ci}-${j}`, section: "Briefs", headline: h,
                            byline: "By STAFF", time: "today", type: "text",
                            summary: "A brief from the newsroom. Follow-up reporting to come.",
                          })}>
                            <span className="tail-bullet">—</span>
                            <span>{h}</span>
                          </li>
                        ))}
                      </ul>
                    </section>
                  );
                }
                return <div key={i}>{item.node}</div>;
              })}
            </div>
          ))}
        </div>

        <footer className="footer">
          <div>© 2026 Daily AI Times · A Modif.AI Publication</div>
          <div>Press tips@dailyaitimes.example · Subscriptions · Corrections</div>
          <div>Keys: <strong>J</strong>/<strong>K</strong> to move · <strong>↵</strong> to open · <strong>Esc</strong> to close</div>
        </footer>
      </div>

      {openStory && (
        <ArticleModal
          story={openStory}
          onClose={() => setOpenStory(null)}
          savedIds={savedIds}
          toggleSave={toggleSave}
        />
      )}

      <Tweaks tweaks={tweaks} setTweaks={setTweaks} visible={tweaksVisible} onClose={() => setTweaksVisible(false)} />
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
