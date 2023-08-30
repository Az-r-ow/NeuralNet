#include <ftxui/dom/elements.hpp>  // for text, gauge, operator|, flex, hbox, Element
#include <ftxui/screen/screen.hpp> // for Screen
#include <iostream>                // for cout, endl, ostream
#include <string>                  // for allocator, char_traits, operator+, operator<<, string, to_string, basic_string
#include <thread>                  // for sleep_for

using namespace ftxui;

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

private:
  int totalIndexes;
  int currIndex;
  std::string preText;
  // string to be printed in order to reset the cursor position
  std::string resetPos;

  void printGauge()
  {
    std::string ratioStr = std::to_string(this->currIndex) + "/" + std::to_string(this->totalIndexes);
    float ratio = static_cast<float>(this->currIndex) / this->totalIndexes;
    Element document = hbox({
        text(this->preText),
        gauge(ratio) | flex,
        text(" " + ratioStr),
    });

    auto screen = Screen(100, 1);
    Render(screen, document);
    std::cout << this->resetPos;
    screen.Print();
    this->resetPos = screen.ResetPosition();
  };
};