;; Blessing and curse of dimensionality

(ns index
  (:require [nextjournal.clerk :as clerk]
            #_[nextjournal.clerk.experimental :as cx]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as op :refer [*=]]
            [clojure.core.matrix.linear :as l]))

(m/set-current-implementation :vectorz)

(def num-vectors 100)

(def vector-len 1000)

(defn rand-unit
  []
  (- (rand 2) 1.0))

(defn random [x y]
  (m/matrix (repeatedly y (partial repeatedly x (partial rand-unit)))))

(def matrix
  (random vector-len num-vectors))

(def norm
  (map #(l/norm % 2) matrix))

(def matrix
  (map #(m/div %1 %2) matrix norm))

(def dot-products
  (m/mmul matrix (m/transpose matrix)))

(def norms
  (m/sqrt (m/diagonal dot-products)))

(def normed-dot-products
  (op// dot-products (m/outer-product norms norms)))

(def angles-degrees
  (->> (apply concat (m/to-degrees (m/acos normed-dot-products)))
       (filter pos?)))

(clerk/vl
 {:data {:values (map (partial hash-map :x) angles-degrees)}
  :mark "bar"
  :width 400
  :height 300
  :encoding {:x {:bin {:step 0.5} :field :x}
             :y {:aggregate "count"}}})
