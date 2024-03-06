#pragma once

#include <ftxui/dom/elements.hpp>  // for text, gauge, operator|, flex, hbox, Element
#include <ftxui/screen/screen.hpp>  // for Screen
#include <iostream>                 // for cout, endl, ostream
#include <string>  // for allocator, char_traits, operator+, operator<<, string, to_string, basic_string
#include <thread>  // for sleep_for

using namespace ftxui;

namespace NeuralNet {
/**
 * A general purpose simple use gauge.
 * It encompasses the few lines of code to have an operational ftxui::gauge
 */

class Gauge {
 public:
  Gauge(std::string preText, int totalIndexes, int currIndex = 0) {
    this->totalIndexes = totalIndexes;
    this->currIndex = currIndex;
    this->preText = preText;
  };

  Gauge &operator++() {
    this->currIndex++;
    printGauge();
    return *this;
  };

 protected:
  int totalIndexes;
  int currIndex;
  std::string preText;
  // string to be printed in order to reset the cursor position
  std::string resetPos;

 private:
  void printGauge() {
    std::string ratioStr = std::to_string(this->currIndex) + "/" +
                           std::to_string(this->totalIndexes);
    float ratio = static_cast<float>(this->currIndex) / this->totalIndexes;
    Element document = hbox({
        text(this->preText),
        gauge(ratio) | flex,
        text(" " + ratioStr),
    });

    auto screen =
        Screen::Create(Dimension::Fixed(100), Dimension::Fit(document));
    Render(screen, document);
    std::cout << this->resetPos;
    screen.Print();
    this->resetPos = screen.ResetPosition();
  };
};

// todo: Refactor this code to not have duplicates
class TrainingGauge : public Gauge {
 public:
  TrainingGauge(int totalIndexes, int currIndex = 0, int totalEpochs = 10,
                int currEpoch = 0)
      : Gauge("Training : ", totalIndexes, currIndex) {
    this->totalEpochs = totalEpochs;
    this->currEpoch = currEpoch;
  }

  void printWithLoss(double l) {
    ++this->currIndex;
    std::string ratioStr = std::to_string(this->currIndex) + "/" +
                           std::to_string(this->totalIndexes);
    std::string epochStr = "Epoch : " + std::to_string(this->currEpoch) + "/" +
                           std::to_string(this->totalEpochs);
    std::string errorStr = "Loss : " + std::to_string(static_cast<float>(l));
    float ratio = static_cast<float>(this->currIndex) / this->totalIndexes;

    auto screen = gaugeBuilder({hbox({
                                    text(epochStr + " "),
                                    gauge(ratio),
                                    text(" " + ratioStr),
                                }) | flex |
                                    border,
                                text(errorStr) | border});

    std::cout << this->resetPos;
    screen.Print();
    this->resetPos = screen.ResetPosition();
  }

  /**
   * Print with loss and accuracy
   */
  void printWithLAndA(double l, double a) {
    ++this->currIndex;
    std::string ratioStr = std::to_string(this->currIndex) + "/" +
                           std::to_string(this->totalIndexes);
    std::string epochStr = "Epoch : " + std::to_string(this->currEpoch) + "/" +
                           std::to_string(this->totalEpochs);
    std::string errorStr = "Loss : " + std::to_string(static_cast<float>(l));
    std::string accStr = "Accuracy : " + std::to_string(static_cast<float>(a));

    float ratio = static_cast<float>(this->currIndex) / this->totalIndexes;

    auto screen =
        gaugeBuilder({hbox({
                          text(epochStr + " "),
                          gauge(ratio),
                          text(" " + ratioStr),
                      }) | flex |
                          border,
                      text(errorStr) | border, text(accStr) | border});

    std::cout << this->resetPos;
    screen.Print();
    this->resetPos = screen.ResetPosition();
  };

  /**
   * @brief Create a horizontal document with the given elements
   *
   * @param elements The elements to be added to the document
   *
   * @return The screen with the given elements
   */
  Screen gaugeBuilder(Elements elements) {
    Element document = hbox(elements);

    Screen screen = Screen::Create(Dimension::Full(), Dimension::Fit(document));

    Render(screen, document);

    return screen;
  }

 private:
  std::string resetPos;
  int totalEpochs;
  int currEpoch;
};
}  // namespace NeuralNet