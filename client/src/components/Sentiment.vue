<template>
  <v-container v-if="!expandedMode" class="sentiment">
    <v-layout align-center justify-center column fill-height>
      <p class="headline mb-0 text-xs-center">{{ sentiment }}</p>
      <v-btn
        v-if="typeof(this.answer) == 'object'"
        large
        depressed
        color="info"
        @click="expand"
      >Show every sentiment</v-btn>
    </v-layout>
  </v-container>
  <v-container v-else class="sentiment">
    <span class="sentiments" v-for="(sentiment, name) in sentiments" :key="name">
      <p class="headline mb-0 text-xs-center">{{ name }}</p>
      <v-spacer></v-spacer>
      <p class="headline mb-0 text-xs-center">{{ sentiment }}</p>
    </span>
    <v-layout align-center justify-center column fill-height>
      <v-btn large depressed color="info" @click="expand">Close</v-btn>
    </v-layout>
  </v-container>
</template>

<script>
const names = {
  emb_cnn_lstm: "CNN with Embeddings and LSTM",
  tfidf_nb: "Naive Bayes with TFiDF Vectorizer",
  tfidf_svc: "SVC with TFiDF Vectorizer",
  w2v_cnn: "CNN with Word2Vec Vectorizer"
};
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
        if (this.answer.emb_cnn_lstm == this.answer.w2v_cnn) {
          final_sentiment[this.answer.emb_cnn_lstm] += 2;
        }
        final_sentiment =
          final_sentiment[0] > final_sentiment[1]
            ? "Tweet is Negative 😟"
            : "Tweet is Positive 😊";
      } else {
        final_sentiment = this.answer;
      }
      return final_sentiment;
    },
    sentiments() {
      let sentiments = this.answer;
      if (typeof this.answer === "object") {
        sentiments = {};
        for (const model in this.answer) {
          let answer = this.answer[model] == 0 ? "Negative 😟" : "Positive 😊";
          sentiments[names[model]] = answer;
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
.sentiment {
  margin-top: -7%;
}
.sentiments {
  display: flex;
  align-content: space-between;
  justify-content: center;
  flex-direction: row;
}
</style>