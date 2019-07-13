<template>
  <v-app>
    <v-toolbar app>
      <v-toolbar-title class="headline text-uppercase">
        <span>Tweet Analyzer</span>
      </v-toolbar-title>
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
