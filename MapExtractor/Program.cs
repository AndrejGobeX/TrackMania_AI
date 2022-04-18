using System;
using GBX.NET;
using GBX.NET.Engines.Game;
using GBX.NET.LZO;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace MapExtractor
{
    class Program
    {
        static void Main(string[] args)
        {
            List<Tuple<List<Tuple<double, double>>, List<Tuple<double, double>>>> arr = new List<Tuple<List<Tuple<double, double>>, List<Tuple<double, double>>>>();
            using StreamWriter file = new(args[0].Substring(0, args[0].Length - 3)+"txt");
            var map = GameBox.ParseNode<CGameCtnChallenge>(args[0]);
            foreach(CGameCtnBlock i in map.Blocks){
                /*Console.WriteLine(i.Name);
                Console.WriteLine(i.AbsolutePositionInMap);
                Console.WriteLine(i.Coord);
                Console.WriteLine(i.Direction);*/

                List<Tuple<double, double>> arr_l = new List<Tuple<double, double>>();
                List<Tuple<double, double>> arr_r = new List<Tuple<double, double>>();
                Vec3 xyz = i.AbsolutePositionInMap;

                if(i.Name == "RoadTechStart" || i.Name == "RoadTechFinish" || i.Name == "RoadTechStraight" || i.Name == "RoadTechCheckpoint")
                {
                    if(i.Direction.ToString() == "North")
                    {
                        arr_r.Add(new Tuple<double, double>(xyz.X, xyz.Z));
                        arr_r.Add(new Tuple<double, double>(xyz.X, xyz.Z + 32));
                        arr_l.Add(new Tuple<double, double>(xyz.X + 32, xyz.Z));
                        arr_l.Add(new Tuple<double, double>(xyz.X + 32, xyz.Z + 32));
                    }
                    else if(i.Direction.ToString() == "South")
                    {
                        arr_l.Add(new Tuple<double, double>(xyz.X, xyz.Z + 32));
                        arr_l.Add(new Tuple<double, double>(xyz.X, xyz.Z));
                        arr_r.Add(new Tuple<double, double>(xyz.X + 32, xyz.Z + 32));
                        arr_r.Add(new Tuple<double, double>(xyz.X + 32, xyz.Z));
                    }
                    else if(i.Direction.ToString() == "East")
                    {
                        arr_r.Add(new Tuple<double, double>(xyz.X + 32, xyz.Z));
                        arr_r.Add(new Tuple<double, double>(xyz.X, xyz.Z));
                        arr_l.Add(new Tuple<double, double>(xyz.X + 32, xyz.Z + 32));
                        arr_l.Add(new Tuple<double, double>(xyz.X, xyz.Z + 32));
                    }
                    else
                    {
                        arr_l.Add(new Tuple<double, double>(xyz.X, xyz.Z));
                        arr_l.Add(new Tuple<double, double>(xyz.X + 32, xyz.Z));
                        arr_r.Add(new Tuple<double, double>(xyz.X, xyz.Z + 32));
                        arr_r.Add(new Tuple<double, double>(xyz.X + 32, xyz.Z + 32));
                    }
                }

                else if(i.Name.Substring(0, 13) == "RoadTechCurve")
                {
                    int r = 1;
                    if(i.Name.IndexOf("2") != -1)
                    {
                        r = 2;
                    }
                    else if(i.Name.IndexOf("3") != -1)
                    {
                        r = 3;
                    }

                    Tuple<double, double> center;
                    double q;
                    if(i.Direction.ToString() == "North")
                    {
                        center = new Tuple<double, double>(xyz.X, xyz.Z + 32 * r);
                        q = 1.5;
                    }
                    else if(i.Direction.ToString() == "East")
                    {
                        center = new Tuple<double, double>(xyz.X, xyz.Z);
                        q = 0;
                    }
                    else if(i.Direction.ToString() == "South")
                    {
                        center = new Tuple<double, double>(xyz.X + 32 * r, xyz.Z);
                        q = 0.5;
                    }
                    else
                    {
                        center = new Tuple<double, double>(xyz.X + 32 * r, xyz.Z + 32 * r);
                        q = 1;
                    }
                    double theta = 0.0;
                    while(theta <= 0.5)
                    {
                        double x = center.Item1 + r * 32 * Math.Cos((theta + q) * Math.PI);
                        double y = center.Item2 + r * 32 * Math.Sin((theta + q) * Math.PI);
                        double x1 = center.Item1 + (r-1) * 32 * Math.Cos((theta + q) * Math.PI);
                        double y1 = center.Item2 + (r-1) * 32 * Math.Sin((theta + q) * Math.PI);
                        arr_l.Add(new Tuple<double, double>(x, y));
                        arr_r.Add(new Tuple<double, double>(x1, y1));
                        theta += 0.05;
                    }
                }

                if(i.Name == "RoadTechStart")
                    arr.Insert(0, new Tuple<List<Tuple<double, double>>, List<Tuple<double, double>>>(arr_r, arr_l));
                else
                    arr.Add(new Tuple<List<Tuple<double, double>>, List<Tuple<double, double>>>(arr_r, arr_l));
            }
            for(int i = 0; i < arr.Count; ++i)
            {
                var current = arr[i];
                var last = current.Item1.Count - 1;
                for (int j = i + 1; j < arr.Count; ++j)
                {
                    var next = arr[j];
                    var nextlast = next.Item2.Count - 1;
                    if(next.Item1[0].Item1 == current.Item1[last].Item1 &&
                        next.Item1[0].Item2 == current.Item1[last].Item2)
                    {
                        var v = arr[i + 1];
                        arr[i + 1] = arr[j];
                        arr[j] = v;
                        break;
                    }
                    else if(next.Item2[nextlast].Item1 == current.Item1[last].Item1 &&
                        next.Item2[nextlast].Item2 == current.Item1[last].Item2)
                    {
                        next.Item1.Reverse();
                        next.Item2.Reverse();
                        var neu = new Tuple<List<Tuple<double, double>>, List<Tuple<double, double>>>(next.Item2, next.Item1);
                        arr[j] = arr[i + 1];
                        arr[i + 1] = neu;
                        break;
                    }
                }
            }
            foreach(var block in arr)
            {
                file.WriteLine("-");
                for(int i = 0; i < block.Item1.Count; ++i)
                {
                    file.WriteLine("" + block.Item1[i].Item1 + " " + block.Item1[i].Item2 + " " + block.Item2[i].Item1 + " " + block.Item2[i].Item2);
                }
            }
            /* file.WriteLine("-");
            foreach(var pair in arr_r)
            {
                file.WriteLine("" + pair.Item1 + " " + pair.Item2);
            }
            file.WriteLine("-");
            foreach(var pair in arr_l)
            {
                file.WriteLine("" + pair.Item1 + " " + pair.Item2);
            } */
        }
    }
}
