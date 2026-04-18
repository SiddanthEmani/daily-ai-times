// Tweaks panel — floating bottom-right controls

function Tweaks({ tweaks, setTweaks, visible, onClose }) {
  if (!visible) return null;
  const set = (k, v) => {
    const next = { ...tweaks, [k]: v };
    setTweaks(next);
    window.parent.postMessage({ type: '__edit_mode_set_keys', edits: { [k]: v } }, '*');
  };
  const Group = ({ label, k, opts }) => (
    <>
      <label>{label}</label>
      <div className="opts">
        {opts.map(o => (
          <button key={o.v} className={tweaks[k] === o.v ? "on" : ""} onClick={() => set(k, o.v)}>
            {o.l}
          </button>
        ))}
      </div>
    </>
  );
  return (
    <div className="tweaks">
      <button className="close" onClick={onClose}>×</button>
      <h4>Tweaks</h4>
      <Group label="Density" k="density" opts={[{v:"comfortable", l:"Comfy"},{v:"compact", l:"Compact"}]} />
      <Group label="Paper tone" k="tone" opts={[{v:"cream", l:"Cream"},{v:"white", l:"White"},{v:"sepia", l:"Sepia"}]} />
      <Group label="Photos" k="photos" opts={[{v:"on", l:"On"},{v:"off", l:"Off"}]} />
      <Group label="Ticker" k="ticker" opts={[{v:"on", l:"On"},{v:"off", l:"Off"}]} />
    </div>
  );
}

Object.assign(window, { Tweaks });
