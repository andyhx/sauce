#!/usr/bin/env escript

main([ConfigFile]) ->
  {ok, Config} = file:consult(ConfigFile),

  {width, Width} = lists:keyfind(width, 1, Config),
  {height, Height} = lists:Keyfind(height, 1, Config),
  {width_stride, WidthStride} = lists:keyfind(width_stride, 1, Config),
  {height_stride, HeightStride} = lists:keyfind(height_stride, 1, Config),

  {method, Method} = lists:keyfind(method, 1, Config),
  MethodC = build_method_command(Method),

  {directory, Directory} = lists:keyfind(directory, 1, Config),
  os:cmd("mkdir " ++ Directory ++ "|| mkdir " ++ Directory ++ "/fps"),

  Prefix = Directory ++ "/" ++ io_lib:format("~p", [Method]),

  {training_pos, TrainingPos} = lists:keyfind(training_pos, 1, Config),
  {training_neg, TrainingNeg} = lists:keyfind(training_neg, 1, Config),
  ok = extract(MethodC, TrainingPos, Prefix ++ "_pos", 100),
  ok = extract(MethodC, TrainingNeg, Prefix ++ "_neg", 100),

  ok = train(Prefix ++ "_pos", Prefix ++ "_neg", Prefix ++ "_svm"),

  {test_pos, TestPos} = lists:keyfind(test_pos, 1, Config),
  {test_neg, TestNeg} = lists:keyfind(test_neg, 1, Config),
  ok = test(pos, MethodC, TestPos, Prefix ++ "_svm"),
  ok = test(neg, MethodC, TestNeg, Prefix ++ "_svm"),

  {training_fp, TrainingFp} = lists:keyfind(training_fp, 1, Config),
  ok = fp(Method, TrainingFp, Directory ++ "/fps", Prefix ++ "_svm", Width, Height),

  {fp_probability, FpProbability} = lists:keyfind(fp_probability, 1, Config),
  ok = extract(MethodC, Directory ++ "/fps", Prefix ++ "_fp", FpProbability),
  ok = join(Prefix ++ "_neg", Prefix ++ "_fp", Prefix ++ "_neg_with_fp"),

  {test_pos, TestPos} = lists:keyfind(test_pos, 1, Config),
  {test_neg, TestNeg} = lists:keyfind(test_neg, 1, Config),
  ok = train(Prefix ++ "_pos", Prefix ++ "_neg_with_fp", Prefix ++ "_svm_retrained"),
  ok = test(pos, MethodC, TestPos, Prefix ++ "_svm_retrained"),
  ok = test(neg, MethodC, TestNeg, Prefix ++ "_svm_retrained"),

  {test_set, TestSet} = lists:keyfind(test_set, 1, Config),
  {test_annotations, TestAnnotations} = lists:keyfind(test_annotations, 1, Config),
  ok = detect(MethodC, TestSet, TestAnnotations, Prefix ++ "_svm_retrained", Prefix ++ "_det", Width, Height, WidthStride, HeightStride);

main([]) ->
  main(["auto.conf"]).


extract(Method, InputSet, OutputFile, Probability) ->
  Command = build_command("extract " ++ InputSet, [{"m", Method},
                                                   {"o", OutputFile},
                                                   {"p", integer_to_list(Probability)}]),

  Count = os:cmd("ls " ++ InputSet ++ " | wc -l"),
  "\n" ++ RevCount = lists:reverse(Count),
  CountI = list_to_integer(lists:reverse(RevCount)),
  io:format("Feature extraction with probability ~p on ~p samples:~n"
            "~p ------[~p]-----> ~p~n", [Probability, CountI, InputSet, Method, OutputFile]),
  {Time, []} = timer:tc(fun() -> os:cmd(Command) end),
  io:format("Total:\t\t~p seconds~nAverage:\t~p ms/sample~n", [Time / 1000000, Time / (10 * Probability * CountI)]),
  ok.

train(Positives, Negatives, Output) ->
  Command = build_command("train", [{"o", Output},
                                    {"p", Positives},
                                    {"n", Negatives}]),

  io:format("Training with linear kernel started~n", []),
  {Time, []} = timer:tc(fun() -> os:cmd(Command) end),
  io:format("Total:\t~p seconds~n", [Time / 1000000]),
  ok.


test(Set, Method, InputSet, Classifier) ->
  io:format("Started cross-validation for ~p set on ~p~n",[Set, InputSet]),
  {Time, {Intended, Total}} = timer:tc(fun() -> do_test(Set, Method, InputSet, Classifier) end),
  io:format("Score:\t~p/~p (~p%)~n", [Intended, Total, Intended*100/Total]),
  io:format("Total:\t~p seconds~n", [Time / 1000000]),
  ok.

do_test(Set, Method, InputSet, Classifier) ->
  Command = build_command("test", [{"s", InputSet},
                                   {"c", Classifier},
                                   {"m", Method}]),

  Res = os:cmd(Command),
  [Zeros, Ones, Total | _] = re:split(Res, "\n", [{return, list}]),
  case Set of
    pos -> {list_to_integer(Ones), list_to_integer(Total)};
    _ -> {list_to_integer(Zeros), list_to_integer(Total)}
  end.

fp(Method, InputSet, OutputDirectory, Classifier, Width, Height) ->
  Command = build_command("fp", [{"i", InputSet},
                                 {"o", OutputDirectory},
                                 {"c", Classifier},
                                 {"w", Width},
                                 {"h", Height}]),

  io:format("Started extracting false-positives~n~p -----> ~p~n",[InputSet, OutputDirectory]),
  {Time, []} = timer:tc(fun() -> os:cmd(Command) end),
  Number = os:cmd("ls " ++ OutputDirectory ++ " | wc -l"),
  "\n" ++ NumberRev = lists:reverse(Number),
  io:format("Generated:\t~p~n", [list_to_integer(lists:reverse(NumberRev))]),
  io:format("Total:\t\t~p~n", [Time / 1000000]),
  ok.

join(Set1, Set2, OutputSet) ->
  Command = build_command("join_sets", [{"a", Set1},
                                        {"b", Set2},
                                        {"o", OutputSet}]),

  io:format("Started joining sets ~p and ~p to ~p~n", [Set1, Set2, OutputSet]),
  {Time, []} = timer:tc(fun() -> os:cmd(Command) end),
  io:format("Total:\t~p~n", [Time / 1000000]),
  ok.

detect(Method, InputSet, Annotations, Classifier, Output, Width, Height, WidthStride, HeightStride) ->
  Command = build_command("detect", [{"i", InputSet},
                                     {"a", Annotations},
                                     {"m", Method},
                                     {"c", Classifier},
                                     {"o", Output},
                                     {"w", Width},
                                     {"h", Height},
                                     {"x", WidthStride},
                                     {"y", HeightStride}]),
  io:format("Started detection on set ~p with annotations ~p, output will be saved to ~p~n", [InputSet, Annotations, Output]),
  {Time, []} = timer:tc(fun() -> os:cmd(Command) end),
  io:format("Total:\t~p~n", [Time / 1000000]),
  ok.

build_command(Command, Parameters) ->
  lists:foldl(fun({Param, Value}, Acc) ->
          Acc ++ " -" ++ Param ++ " " ++ Value
      end, "./sauce" ++ Command, Parameters).

build_method_command({Method, Parameters}) when is_list(Parameters) ->
  lists:foldl(fun({Param, Value}, Acc) ->
          Acc ++ " -" ++ Param ++ " " ++ Value
      end, Method, Parameters);

build_method_command(Method) ->
  Method.
