<template>
  <v-app>
    <v-toolbar app flat>
      <v-layout align-center justify-center row fill-height>
        <v-toolbar-title>
          <span>Tweet Analyzer</span>
        </v-toolbar-title>
      </v-layout>
    </v-toolbar>
    <v-spacer></v-spacer>
    <Analyzer />
    <Footer />
  </v-app>
</template>

<script>
import Footer from "./components/Footer";
import Analyzer from "./components/Analyzer";

export default {
  name: "App",
  components: {
    Footer,
    Analyzer
  },
  data() {
    return {
      //
    };
  }
};
</script>

<style scoped>
span {
  font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans;
  font-size: 3.3em;
}
.v-toolbar {
  background-color: #fafafa;
}
</style>
