(defproject warren "0.1.0-SNAPSHOT"
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/core.matrix "0.57.0"]
                 [net.mikera/vectorz-clj "0.45.0"]
                 [synaptic "0.2.0"]]

  :main ^:skip-aot warren.core

  :profiles {:dev {:dependencies [[expectations "2.1.8"]]
                   :plugins [[lein-expectations "0.0.8"]
                             [lein-autoexpect "1.9.0"]]}
             :uberjar {:aot :all
                       :uberjar-name "warren.jar"}})
