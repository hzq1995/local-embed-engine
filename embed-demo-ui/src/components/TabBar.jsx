/**
 * TabBar — top-level scene switcher.
 *
 * Props:
 *   scenes: Array<{ id: string, label: string }>
 *   activeScene: string
 *   onSwitch: (id: string) => void
 */
export default function TabBar({ scenes, activeScene, onSwitch }) {
  return (
    <div className="tab-bar">
      {scenes.map((scene) => (
        <button
          key={scene.id}
          type="button"
          className={`tab-btn${activeScene === scene.id ? " active" : ""}`}
          onClick={() => onSwitch(scene.id)}
        >
          {scene.label}
        </button>
      ))}
    </div>
  );
}
