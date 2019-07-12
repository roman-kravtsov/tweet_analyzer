<template>
  <v-app>
    <v-toolbar app>
      <v-toolbar-title class="headline text-uppercase">
        <span>Tweet Analyzer</span>
      </v-toolbar-title>
      <v-spacer />
    </v-toolbar>
    <v-spacer/>
    <Introduction />
    <Analyzer />
    <Footer />
  </v-app>
</template>

<script>
import Footer from './components/Footer'
import Analyzer from './components/Analyzer'
import Introduction from './components/Introduction'

export default {
  name: 'App',
  components: {
    Footer,
    Analyzer,
    Introduction
  },
  data () {
    return {
      //
    }
  }
}
</script>
