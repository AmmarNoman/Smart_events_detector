var circleAnimation = anime({
  targets: ".img-table_wrapper.loading > span",
  scale: ["0.8", "1.4"],
  opacity: ["1", 0],
  easing: "easeOutCubic",
  duration: 1500,
  loop: true
});

var pointAnimation = anime({
  targets: ".point-animated",
  translateX: [
    { value: 5, duration: 1, delay: 499 },
    { value: 10, duration: 1, delay: 499 },
    { value: 15, duration: 1, delay: 499 }
  ],
  loop: true,
  easing: "easeOutCubic"
});

var pauseButton = document.querySelector(".pause");
var restartButton = document.querySelector(".restart");

pauseButton.addEventListener("click", function() {
  circleAnimation.pause();
  pointAnimation.pause();
});
restartButton.addEventListener("click", function() {
  circleAnimation.restart();
  pointAnimation.restart();
});