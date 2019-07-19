<template>
  <v-container v-if="!expandedMode">
    <p class="headline mb-0 text-xs-center">{{sentiment}}</p>
    <v-btn large depressed color="info" class="centered" @click="expand">Show every sentiment</v-btn>
  </v-container>
  <v-container v-else>
    <p
      v-for="sentiment in sentiments"
      :key="sentiment"
      class="headline mb-0 text-xs-center"
    >{{ sentiment }}</p>
    <v-btn large depressed color="info" class="centered" @click="expand">Close</v-btn>
  </v-container>
</template>

<script>
export default {
  name: "Sentiment",
  props: {
    answer: {
      required: true
    }
  },
  data() {
    return {
      expanded: false
    };
  },
  computed: {
    expandedMode() {
      return this.expanded && typeof this.answer == "object";
    },
    sentiment() {
      let final_sentiment = { 0: 0, 1: 0 };
      if (typeof this.answer === "object") {
        for (const sentiment of Object.values(this.answer)) {
          final_sentiment[sentiment] += 1;
        }
        final_sentiment =
          final_sentiment[0] > final_sentiment[1] ? "Tweet is Negative😟" : "Tweet is Positive😊";
      } else {
        final_sentiment = this.answer;
      }
      return final_sentiment;
    },
    sentiments() {
      let sentiments = this.answer;
      if (typeof this.answer === "object") {
        sentiments = [];
        for (const model in this.answer) {
          let answer = this.answer[model] == 0 ? "Negative😟" : "Positive😊";
          if (this.answer[model] == "Not implemented") {
            answer = "Not implemented";
          }
          sentiments.push(`${model}: ${answer}`);
        }
      }
      return sentiments;
    }
  },
  methods: {
    expand() {
      this.expanded = !this.expanded;
    }
  }
};
</script>

<style >
.centered {
  margin-top: 2%;
  position: relative;
  float: left;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}
button.centered {
  margin-top: 5%;
}
</style>