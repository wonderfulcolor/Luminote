export function createTurnModalController({ dialog, closeButton, render, keyTarget = dialog.ownerDocument || dialog }) {
  let activeSnapshot = null;

  const close = () => {
    if (typeof dialog.close === 'function') dialog.close();
    else dialog.removeAttribute('open');
    activeSnapshot = null;
  };

  const open = snapshot => {
    activeSnapshot = snapshot;
    render(snapshot);
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  };

  closeButton.addEventListener('click', close);
  dialog.addEventListener('click', event => {
    if (event.target === dialog) close();
  });
  dialog.addEventListener('cancel', () => { activeSnapshot = null; });
  keyTarget.addEventListener('keydown', event => {
    if (event.key === 'Escape' && activeSnapshot !== null) {
      event.preventDefault();
      close();
    }
  });

  return {
    open,
    close,
    currentSnapshot: () => activeSnapshot,
  };
}
