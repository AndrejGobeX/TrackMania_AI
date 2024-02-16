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
            using StreamWriter file = new(args[0].Substring(0, args[0].Length - 3)+"txt");
            var map = GameBox.ParseNode<CGameCtnChallenge>(args[0]);
            foreach(CGameCtnBlock i in map.Blocks){
                file.WriteLine(i.Name);
                var coords = new Int3(i.Coord.X*32, i.Coord.Z*32, i.Coord.Y*8-62);
                file.WriteLine(coords);
                file.WriteLine(i.Direction);
            }
        }
    }
}
