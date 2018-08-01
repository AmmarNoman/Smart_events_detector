"use strict";

Vue.component("doughnut", {
  props: ["data", "labels"],
  data: function data() {
    ctx: null;
  },
  template: "\n    <canvas></canvas>\n  ",
  mounted: function mounted() {
    var self = this;
    this.ctx = this.$el.getContext("2d");

    new Chart(this.ctx, {
      type: "doughnut",
      options: {
        cutoutPercentage: 80
      },
      data: {
        labels: self.labels,
        datasets: [{
          data: self.data,
          backgroundColor: ["#1BC98E", "#1CA8DD"],
          hoverBackgroundColor: ["#1BC98E", "#1CA8DD"]
        }]
      }
    });
  }
});

Vue.component("sparkline", {
  props: ["title", "value"],
  data: function data() {
    ctx: null;
  },
  template: "\n    <div class=\"br2\">\n      <div class=\"pa3 flex-auto bb b--white-10\">\n        <h3 class=\"mt0 mb1 f6 ttu white o-70\">{{ title }}</h3>\n        <h2 class=\"mv0 f2 fw5 white\">{{ value }}</h2>\n      </div>\n      <div class=\"pt2\">\n        <canvas></canvas>\n      </div>\n    </div>\n  ",
  mounted: function mounted() {
    this.ctx = this.$el.querySelector("canvas").getContext("2d");
    var sparklineGradient = this.ctx.createLinearGradient(0, 0, 0, 135);
    sparklineGradient.addColorStop(0, "rgba(255,255,255,0.35)");
    sparklineGradient.addColorStop(1, "rgba(255,255,255,0)");

    var data = {
      labels: ["A", "B", "C", "D", "E", "F"],
      datasets: [{
        backgroundColor: sparklineGradient,
        borderColor: "#FFFFFF",
        data: [2, 4, 6, 4, 8, 10]
      }]
    };

    Chart.Line(this.ctx, {
      data: data,
      options: {
        elements: {
          point: {
            radius: 0
          }
        },
        scales: {
          xAxes: [{
            display: false
          }],
          yAxes: [{
            display: false
          }]
        }
      }
    });
  }
});

Vue.component("metric-list-item", {
  props: ["name", "value", "showBar"],
  computed: {
    barWidth: function barWidth() {
      return this.value + "%";
    }
  },
  template: "\n    <a href=\"#\" class=\"link dark-gray flex justify-between relative pa3 bb b--black-10 hover-bg-near-white\">\n      <span v-if=\"showBar\" class=\"absolute top-0 left-0 right-0 bottom-0 h-100 bg-near-white\" v-bind:style=\"{ width: barWidth, zIndex: -1 }\"></span>\n      <span>{{ name }}</span>\n      <span>{{ value }}</span>\n    </a>\n  "
});

