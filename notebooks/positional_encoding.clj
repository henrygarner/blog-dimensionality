(ns positional-encoding
  (:require [nextjournal.clerk :as clerk]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as op :refer [*=]]
            [clojure.core.matrix.linear :as l]))

(defn synthetic-data
  [nx ny]
  (for [x (range nx) y (range ny)]
    {:x x :y y :temp (rand)}))

(def data
  (synthetic-data 250 250))

(clerk/vl
 {:data {:values data},
  :title "Example heatmap",
  :config {:view {:strokeWidth 0, :step 2},
           :axis {:domain false}},
  :mark "rect",
  :encoding
  {:x
   {:field "x",
    :type "ordinal"},
   :y
   {:field "y",
    :type "ordinal"},
   :color
   {:field "temp",
    :type "quantitative", 
    :legend {:title "Example legend"}}}})
