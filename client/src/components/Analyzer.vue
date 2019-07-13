<template>
  <v-content>
    <v-flex xs12 sm6 offset-sm3>
      <Introduction />
      <v-container>
        <v-text-field
          v-model="text"
          outline
          label="Tweet"
          placeholder="Plese type in your future tweet"
          prepend-icon="fab fa-twitter"
          height="100px"
          @blur="disableLoading()"
        ></v-text-field>
      </v-container>
      <v-container>
        <v-card class="analyzer">
          <v-card-title primary-title>
            <div>
              <h3 class="headline mb-0 text-xs-center">{{ answer }}</h3>
            </div>
          </v-card-title>
        </v-card>
        <!-- <v-progress-circular
          v-if="isLoading"
          class="analyzer"
          indeterminate
          color="primary"
          size="96"
        ></v-progress-circular>-->
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
    this.debouncedGetSentiment = debounce(this.getSentiment, 1000);
  },
  methods: {
    async getSentiment() {
      if (this.text.length === 0) {
        this.answer = "Nothing to analyze";
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
    },
    enableLoading() {
      this.isLoading = true;
    },
    disableLoading() {
      this.isLoading = false;
    }
  }
};
</script>

<style scoped>
.analyzer {
  position: relative;
  float: left;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  /* padding: 0px; */
}
</style>
