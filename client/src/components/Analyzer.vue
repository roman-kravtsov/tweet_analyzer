<template>
  <v-content>
    <v-flex xs12 sm6 offset-sm3>
      <Introduction />
      <v-container>
        <v-textarea
          v-model="text"
          outline
          label="Tweet"
          placeholder="Plese type in your future tweet"
          append-icon="fab fa-twitter"
          @blur="disableLoading()"
        ></v-textarea>
      </v-container>
      <v-container v-if="isLoading">
        <v-progress-circular class="centered" indeterminate color="primary" size="72"></v-progress-circular>
      </v-container>
      <v-container v-if="!isLoading" class="centered">
        <div>
          <h3 class="headline mb-0 text-xs-center">{{ answer }}</h3>
        </div>
      </v-container>
    </v-flex>
  </v-content>
</template>

<script>
import { debounce } from "lodash";
import Introduction from "./Introduction";

export default {
  name: "Analyzer",
  components: {
    Introduction
  },
  data() {
    return {
      text: "",
      isLoading: false,
      answer: "Nothing to analyze"
    };
  },
  watch: {
    text() {
      this.answer = "Waiting until you stop typing...";
      this.debouncedGetSentiment();
    }
  },
  created() {
    this.debouncedGetSentiment = debounce(this.getSentiment, 450);
  },
  methods: {
    async getSentiment() {
      if (this.text.length === 0) {
        this.answer = "Nothing to analyze";
        return;
      }
      this.isLoading = true;
      await new Promise(resolve => setTimeout(resolve, 500));
      const vm = this;
      fetch("http://127.0.0.1:5000/predict", {
        method: "post",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ tweet: vm.text })
      })
        .then(response => response.json())
        .then(response => {
          vm.isLoading = false;
          vm.answer = response;
        })
        .catch(function(error) {
          vm.isLoading = false;
          vm.answer = `Error! ${error}`;
        });
    },
    disableLoading() {
      this.isLoading = false;
    }
  }
};
</script>

<style scoped>
.centered {
  position: relative;
  float: left;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}
</style>
