export function isNearBottom(scroller, threshold = 96) {
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= threshold;
}
