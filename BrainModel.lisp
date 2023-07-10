(defpackage :brain-model
  (:use :common-lisp)
  (:export :add-connection :fire-neuron))

(in-package :brain-model)

(defstruct neuron
  "A simple structure representing a neuron."
  (connections '() :type list))

(defun add-connection (neuron target)
  "Add a synaptic connection from NEURON to TARGET."
  (push target (neuron-connections neuron)))

(defun fire-neuron (neuron)
  "Fire a NEURON, stimulating all neurons it is connected to."
  (dolist (target (neuron-connections neuron))
    (format t "Neuron ~A fires and stimulates neuron ~A~%" neuron target)))

(defun main ()
  ;; Create two neurons
  (let ((neuron1 (make-neuron))
        (neuron2 (make-neuron)))
    ;; Connect neuron1 to neuron2
    (add-connection neuron1 neuron2)
    ;; Fire neuron1, which will stimulate neuron2
    (fire-neuron neuron1)))

(main)
