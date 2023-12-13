speaker_fullname = {"PM": "product manager",
                    "ME":"marketing expert",
                    "UI":"interface designer",
                    "ID":"industrial designer"}
def resolve_first_pronouns(sentence, speaker_name):
    # Split the sentence into words
    words = sentence.split()

    # Iterate and replace pronouns
    for i in range(len(words)):
        if words[i].lower() == "we" or words[i].lower() == "our" or words[i] == "us":
            words[i] = "our team"

        elif words[i].lower() == "i" or words[i].lower() == "my" or words[i]==speaker_name:
            words[i] = speaker_fullname[speaker_name]

    # Join the words back into a sentence
    modified_sentence = ' '.join(words)
    return modified_sentence

def get_span_words(span, document):
    return " ".join(document[span[0] : span[1] + 1])

def resolve_pronouns(prediction):
  text = " ".join(prediction['document']).lower()

  document = prediction['document']
  clusters = prediction['clusters']
  span_to_rep_mention = {}
  for cluster in clusters:
      # The first span in each cluster is considered the representative mention
      rep_mention = get_span_words(cluster[0], document)
      for span in cluster:
          span_to_rep_mention[tuple(span)] = rep_mention
  sorted_spans = sorted([span for cluster in clusters for span in cluster], key=lambda x: x[0])
  # Initialize a list to hold the new tokens
  new_tokens = []
  last_end = 0
  for span in sorted_spans:
      # Get the start and end of the current span
      start, end = span
      # Append the text from the last end to the current start
      new_tokens.extend(document[last_end:start])
      # Append the representative mention for the current span
      new_tokens.append(span_to_rep_mention[tuple(span)])
      # Update the last end
      last_end = end + 1
  # Append any remaining text after the last span
  new_tokens.extend(document[last_end:])
  resolved_text = " ".join(new_tokens[2:])
  speaker = new_tokens[0]

  resolved_text = resolve_first_pronouns(resolved_text,speaker)
  return resolved_text
