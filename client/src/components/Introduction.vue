<template>
  <v-container class="tweet">
    <v-icon medium left>fab fa-twitter-square</v-icon>
    <div class="josefin">
      Welcome to
      <strong>Tweet Analyzer</strong>. This fancy
      application will analyze the sentiment of your tweet. Please type
      in your future tweet below.
    </div>
    <v-list-tile class="grow">
      <v-layout align-center justify-end>
        <v-icon class="mr-1">far fa-heart</v-icon>
        <span class="subheading mr-2">256</span>
        <span class="mr-1">·</span>
        <v-icon class="mr-1">fas fa-retweet</v-icon>
        <span class="subheading">45</span>
      </v-layout>
    </v-list-tile>
  </v-container>
</template>

<script>
export default {
  name: "Introduction",
  data: () => ({
    //
  })
};
</script>

<style>
.tweet {
  background-color: rgb(173, 214, 253);
  border-radius: 25px;
}
.josefin {
  font-size: 18px !important;
  font-weight: 400;
  line-height: 32px !important;
  letter-spacing: normal;
  font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans-serif;
}
</style>
