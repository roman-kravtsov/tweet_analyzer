<template>
  <v-content>
    <v-container class="analyzer">
      <v-text-field v-model="text" outline label="Put your tweet" prepend-icon="fab fa-twitter"></v-text-field>
    </v-container>
  </v-content>
</template>

<script>
export default {
  name: 'Analyzer',
  data () {
    return {
      text: ''
    }
  }
}
</script>

<style>

.analyzer {
  position: relative;
  float: left;
  /* top: 50%; */
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 30px;
}
</style>
