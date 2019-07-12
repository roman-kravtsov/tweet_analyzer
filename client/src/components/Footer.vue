<template>
  <v-footer color="primary lighten-1" padless :absolute="true">
    <v-card class="flex" flat tile>
      <v-card-title class="blue-grey lighten-2">
        <strong class="subheading">Get connected with us on social networks!</strong>
        <v-spacer></v-spacer>
        <v-btn v-for="icon in icons" :key="icon" class="mx-3" dark icon>
          <v-icon size="24px">{{ icon }}</v-icon>
        </v-btn>
      </v-card-title>
    </v-card>
  </v-footer>
</template>

<script>
export default {
  name: 'Footer',
  data: () => ({
    icons: ['fab fa-github']
  })
}
</script>
