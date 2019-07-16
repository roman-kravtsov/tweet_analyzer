<template>
  <v-footer height="5%" :app="true">
    <v-layout align-center justify-center row fill-height>
      <v-btn href="https://github.com/Chewbaccamaster/tweet_analyzer" class="mb-2" light icon>
        <v-icon size="24px">{{ "fab fa-github" }}</v-icon>
      </v-btn>
    </v-layout>
  </v-footer>
</template>

<script>
export default {
  name: 'Footer',
  data: () => ({})
}
</script>
