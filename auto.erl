#!/usr/bin/env escript

main([ConfigFile]) ->
  {ok, Config} = file:consult(ConfigFile),
  {method, Method} = lists:keyfind(method, 1, Config),
  {directory, Directory} = lists:keyfind(directory, 1, Config),
  os:cmd("mkdir " ++ Directory ++ "|| mkdir " ++ Directory ++ "/fps"),

  Prefix = Directory ++ "/" ++ Method,

%%   {training_pos, TrainingPos} = lists:keyfind(training_pos, 1, Config),
%%   {training_neg, TrainingNeg} = lists:keyfind(training_neg, 1, Config),
%%   ok = extract(Method, TrainingPos, Prefix ++ "_pos", 100),
%%   ok = extract(Method, TrainingNeg, Prefix ++ "_neg", 100),
%% 
%%   ok = train(Prefix ++ "_pos", Prefix ++ "_neg", Prefix ++ "_svm"),
%% 
%%   {test_pos, TestPos} = lists:keyfind(test_pos, 1, Config),
%%   {test_neg, TestNeg} = lists:keyfind(test_neg, 1, Config),
%%   ok = test(pos, Method, TestPos, Prefix ++ "_svm"),
%%   ok = test(neg, Method, TestNeg, Prefix ++ "_svm"),

%%   {training_fp, TrainingFp} = lists:keyfind(training_fp, 1, Config),
%%   ok = fp(Method, TrainingFp, Directory ++ "/fps", Prefix ++ "_svm"),

%%   {fp_probability, FpProbability} = lists:keyfind(fp_probability, 1, Config),
%%   ok = extract(Method, Directory ++ "/fps", Prefix ++ "_fp", FpProbability),
%%   ok = join(Prefix ++ "_neg", Prefix ++ "_fp", Prefix ++ "_neg_with_fp"),

  {test_pos, TestPos} = lists:keyfind(test_pos, 1, Config),
  {test_neg, TestNeg} = lists:keyfind(test_neg, 1, Config),
  ok = train(Prefix ++ "_pos", Prefix ++ "_neg_with_fp", Prefix ++ "_svm_retrained"),
  ok = test(pos, Method, TestPos, Prefix ++ "_svm_retrained"),
  ok = test(neg, Method, TestNeg, Prefix ++ "_svm_retrained");

main([]) ->
  main(["auto.conf"]).


extract(Method, InputSet, OutputFile, Probability) ->
  Command = "./sauce extract " ++ InputSet ++ " -m " ++ Method ++ " -o " ++ OutputFile ++ " -p " ++ integer_to_list(Probability),
  Count = os:cmd("ls " ++ InputSet ++ " | wc -l"),
  "\n" ++ RevCount = lists:reverse(Count),
  CountI = list_to_integer(lists:reverse(RevCount)),
  io:format("Feature extraction with probability ~p on ~p samples:~n"
            "~p ------[~p]-----> ~p~n", [Probability, CountI, InputSet, Method, OutputFile]),
  {Time, []} = timer:tc(fun() -> os:cmd(Command) end),
  io:format("Total:\t\t~p seconds~nAverage:\t~p ms/sample~n", [Time / 1000000, Time / (10 * Probability * CountI)]),
  ok.

train(Positives, Negatives, Output) ->
  Command = "./sauce train -o " ++ Output ++ " -p " ++ Positives ++ " -n " ++ Negatives,
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
  Command = "./sauce test -s " ++ InputSet ++ " -c " ++ Classifier ++ " -m " ++ Method,
  Res = os:cmd(Command),
  [Zeros, Ones, Total | _] = re:split(Res, "\n", [{return, list}]),
  case Set of
    pos -> {list_to_integer(Ones), list_to_integer(Total)};
    _ -> {list_to_integer(Zeros), list_to_integer(Total)}
  end.

fp(Method, InputSet, OutputDirectory, Classifier) ->
  Command = "./sauce fp -m " ++ Method ++ " -i " ++ InputSet ++ " -o " ++ OutputDirectory ++ " -c " ++ Classifier,
  io:format("Started extracting false-positives~n~p -----> ~p~n",[InputSet, OutputDirectory]),
  {Time, []} = timer:tc(fun() -> os:cmd(Command) end),
  Number = os:cmd("ls " ++ OutputDirectory ++ " | wc -l"),
  "\n" ++ NumberRev = lists:reverse(Number),
  io:format("Generated:\t~p~n", [list_to_integer(lists:reverse(NumberRev))]),
  io:format("Total:\t\t~p~n", [Time / 1000000]),
  ok.

join(Set1, Set2, OutputSet) ->
  Command = "./sauce join_sets -a " ++ Set1 ++ " -b " ++ Set2 ++ " -o " ++ OutputSet,
  io:format("Started joining sets ~p and ~p to ~p~n", [Set1, Set2, OutputSet]),
  {Time, []} = timer:tc(fun() -> os:cmd(Command) end),
  io:format("Total:\t~p~n", [Time / 1000000]),
  ok.
