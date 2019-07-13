<template>
  <v-content>
    <v-container class="analyzer">
      <v-text-field v-model="text" outline label="Put your tweet" prepend-icon="fab fa-twitter"></v-text-field>
    </v-container>
    <v-container class="analyzer">
      <v-layout>
        <v-flex xs12 sm6 offset-sm3>
          <v-card>
            <v-card-title primary-title>
              <div>
                <h3 class="headline mb-0">{{ answer }}</h3>
              </div>
            </v-card-title>
          </v-card>
        </v-flex>
      </v-layout>
      <v-progress-circular
        v-if="isLoading"
        class="analyzer"
        indeterminate
        color="primary"
        size="96"
      ></v-progress-circular>
    </v-container>
  </v-content>
</template>

<script>
import { debounce } from "lodash";

export default {
  name: "Analyzer",
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
    this.debouncedGetSentiment = debounce(this.getSentiment, 1000);
  },
  methods: {
    async getSentiment() {
      if (this.text.length === 0) {
        return;
      }
      this.answer = "Getting sentiment...";
      this.isLoading = true;
      const vm = this;
      const a = await fetch("http://127.0.0.1:5000/predict", {
        method: "post",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ tweet: vm.text })
      })
        .then(response => response.json())
        .then(response => ((vm.isLoading = false), (vm.answer = response)))
        .catch(function(error) {
          vm.isLoading = false;
          vm.answer = "Error! " + error;
        });
    }
  }
};
</script>

<style>
.analyzer {
  position: relative;
  float: left;
  left: 50%;
  transform: translate(-50%, -50%);
}
</style>
