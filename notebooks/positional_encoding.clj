(ns positional-encoding
  (:require [nextjournal.clerk :as clerk]
            [kixi.stats.core :as k]
            [kixi.stats.math :as km]))

(defn make-vectoriser
  [k]
  (let [d (* k 2)]
    (fn [t]
      (->> (for [i (range k)
                 :let [wk (/ 1.0 (km/pow 10000 (/ i k)))
                       wkt (* wk t)]]
             (vector (km/sin wkt)
                     (km/cos wkt)))
           (apply concat)
           (vec)))))

(def vectoriser
  (make-vectoriser 100))

(vectoriser 1)

(vectoriser 100)

(def data
  (for [t (range 1000)
        :let [v (vectoriser t)]
        [i x] (map-indexed vector v)]
    {:x i :y t :val x}))

(clerk/vl
 {:data {:values data},
  :title "Positional encoding for token offsets",
  :config {:view {:strokeWidth 0, :step 3},
           :axis {:domain false}},
  :mark "rect",
  :encoding
  {:x
   {:field "x",
    :type "ordinal"
    :axis {:values (range 0 1000 10)
           :title "Index into positional vector"}},
   :y
   {:field "y",
    :type "ordinal"
    :axis {:values (range 0 1000 10)
           :title "Token offset"}},
   :color
   {:field "val",
    :type "quantitative", 
    :legend {:title "Example legend"}}}})
