// BrainModel.scala
class Neuron {
  private var memory: List[String] = List()
  private var decisionThreshold: Int = 5

  def receiveInput(input: String): Unit = {
    if (shouldRemember(input)) {
      memory ::= input
      updateThreshold()
    }
  }

  private def shouldRemember(input: String): Boolean = input.length > decisionThreshold

  private def updateThreshold(): Unit = {
    decisionThreshold = memory.map(_.length).sum / memory.length
  }

  def retrieveMemory(): List[String] = memory
}

object BrainModel {
  def main(args: Array[String]): Unit = {
    val neuron = new Neuron
    neuron.receiveInput("This is some sensory input")
    neuron.receiveInput("This input will be forgotten")
    println(neuron.retrieveMemory())
  }
}
