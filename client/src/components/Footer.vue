<template>
  <v-footer color="primary lighten-1" padless :app="true">
    <v-card class="flex" flat tile>
      <v-card-title class="teal lighten-3">
        <strong class="subheading">Get connected with us on social networks!</strong>
        <v-spacer />
        <v-btn v-for="icon in icons" :key="icon" class="mb-2" dark icon>
          <v-icon size="36px">{{ icon }}</v-icon>
        </v-btn>
      </v-card-title>
    </v-card>
  </v-footer>
</template>

<script>
export default {
  name: "Footer",
  data: () => ({
    icons: ["fab fa-github"]
  })
};
</script>
