#pragma once

#include <ftxui/dom/elements.hpp>  // for text, gauge, operator|, flex, hbox, Element
#include <ftxui/screen/screen.hpp> // for Screen
#include <iostream>                // for cout, endl, ostream
#include <string>                  // for allocator, char_traits, operator+, operator<<, string, to_string, basic_string
#include <thread>                  // for sleep_for

namespace NeuralNet
{
  using namespace ftxui;

  /**
   * A general purpose simple use gauge.
   * It encompasses the few lines of code to have an operational ftxui::gauge
   */
  class Gauge
  {
  public:
    Gauge(std::string preText, int totalIndexes, int currIndex = 0)
    {
      this->totalIndexes = totalIndexes;
      this->currIndex = currIndex;
      this->preText = preText;
    };

    Gauge &operator++()
    {
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
    void printGauge()
    {
      std::string ratioStr = std::to_string(this->currIndex) + "/" + std::to_string(this->totalIndexes);
      float ratio = static_cast<float>(this->currIndex) / this->totalIndexes;
      Element document = hbox({
          text(this->preText),
          gauge(ratio) | flex,
          text(" " + ratioStr),
      });

      auto screen = Screen::Create(Dimension::Fixed(100), Dimension::Fit(document));
      Render(screen, document);
      std::cout << this->resetPos;
      screen.Print();
      this->resetPos = screen.ResetPosition();
    };
  };

  // todo: Refactor this code to not have duplicates
  class TrainingGauge : public Gauge
  {
  public:
    TrainingGauge(std::string preText, int totalIndexes, int currIndex = 0) : Gauge(preText, totalIndexes, currIndex) {}

    void printWithError(double e)
    {
      ++this->currIndex;
      std::string ratioStr = std::to_string(this->currIndex) + "/" + std::to_string(this->totalIndexes);
      std::string errorStr = "Loss : " + std::to_string(static_cast<float>(e));
      float ratio = static_cast<float>(this->currIndex) / this->totalIndexes;
      Element document = vbox({
          hbox({
              text(this->preText),
              gauge(ratio) | flex,
              text(" " + ratioStr),
          }),
          text(errorStr),
      });

      auto screen = Screen::Create(Dimension::Fixed(100), Dimension::Fit(document));
      Render(screen, document);
      std::cout << this->resetPos;
      screen.Print();
      this->resetPos = screen.ResetPosition();
    };

  private:
    std::string resetPos;
  };
}