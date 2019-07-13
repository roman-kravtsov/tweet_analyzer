<template>
  <v-card class="mx-auto" color="light-blue lighten-3" :elevation="10">
    <v-card-title>
      <v-icon medium left>fab fa-twitter-square</v-icon>
    </v-card-title>
    <v-card-text class="headline">
      Welcome to
      <strong>Tweet Analyzer</strong>. This fancy
      application will analyze the sentiment of your tweet. Please type
      in your future tweet below.
    </v-card-text>
    <v-card-actions>
      <v-list-tile class="grow">
        <v-layout align-center justify-end>
          <v-icon class="mr-1">mdi-heart</v-icon>
          <span class="subheading mr-2">256</span>
          <span class="mr-1">·</span>
          <v-icon class="mr-1">mdi-share-variant</v-icon>
          <span class="subheading">45</span>
        </v-layout>
      </v-list-tile>
    </v-card-actions>
  </v-card>
</template>

<script>
export default {
  name: "Introduction",
  data: () => ({
    //
  })
};
</script>
